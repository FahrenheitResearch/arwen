# tests/test_boundary_strata.py  (WP-5 workstream B: boundary-distance
# stratification of the matched-run comparator)
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOL = (Path(__file__).resolve().parents[1]
         / "tools" / "matched_wrfout_stream_compare.py")


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "matched_wrfout_stream_compare", _TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sc = _load_tool()

#: Cross-check tolerance for interior metrics computed two ways.  The
#: --boundary-strata interior band (d >= 5) and the legacy
#: --exclude-rows 5 interior grid are the SAME cell set, so the two paths
#: differ only by summation order over a permuted selection; the bound is
#: the FP64 accumulation slack that permutation can introduce and nothing
#: larger.  The measured maximum discrepancy is asserted and printed.
INTERIOR_CROSS_CHECK_TOL = 1.0e-12


def _frame(ny, nx, nz, rng):
    return {
        "T2": rng.standard_normal((ny, nx)).astype(np.float32) + 290.0,
        "PSFC": rng.standard_normal((ny, nx)).astype(np.float32) + 98000.0,
        "U10": rng.standard_normal((ny, nx)).astype(np.float32),
        "V10": rng.standard_normal((ny, nx)).astype(np.float32),
        "QVAPOR": rng.random((nz, ny, nx)).astype(np.float32) * 0.01,
        "W": rng.standard_normal((nz, ny, nx)).astype(np.float32),
        "RAINNC": rng.random((ny, nx)).astype(np.float32),
        "REFL_10CM": (rng.random((nz, ny, nx)).astype(np.float32) * 40.0),
    }


def test_boundary_distance_matches_runtime_convention():
    """Cell for cell against the runtime's own attribution expression."""
    ny, nx = 13, 17
    got = sc.boundary_distance(ny, nx)
    expected = np.empty((ny, nx), dtype=got.dtype)
    for j in range(ny):
        for i in range(nx):
            # gpuwm/runtime.py boundary-row attribution, transcribed.
            expected[j, i] = min(j, ny - 1 - j, i, nx - 1 - i)
    np.testing.assert_array_equal(got, expected)


def test_strata_bands_partition_the_grid():
    ny, nx = 24, 30
    width = 5
    bands = dict(sc.strata_masks(ny, nx, width))
    assert bands["d0_specified"].sum() > 0
    assert bands["interior"].sum() > 0
    # the three reported bands tile the grid exactly once
    total = (bands["d0_specified"].astype(int)
             + bands["relaxation"].astype(int)
             + bands["interior"].astype(int))
    np.testing.assert_array_equal(total, np.ones((ny, nx), dtype=int))
    # and the aggregate relaxation band is the union of its rows
    union = np.zeros((ny, nx), dtype=bool)
    for row in range(1, width):
        union |= bands[f"d{row}_relaxation"]
    np.testing.assert_array_equal(union, bands["relaxation"])
    with pytest.raises(ValueError):
        sc.strata_masks(ny, nx, 0)


def test_interior_band_agrees_with_exclude_rows(capsys):
    """One field, two code paths, one committed comparison."""
    rng = np.random.default_rng(20260731)
    ny, nx, nz, width = 26, 32, 4, 5
    gpu = _frame(ny, nx, nz, rng)
    cpu = {k: v + rng.standard_normal(v.shape).astype(np.float32) * 0.01
           for k, v in gpu.items()}

    interior_mask = dict(sc.strata_masks(ny, nx, width))["interior"]
    banded = sc.compare_masked(gpu, cpu, interior_mask)
    trimmed = sc._compare_fields(
        {k: sc._interior(v, width) for k, v in gpu.items()},
        {k: sc._interior(v, width) for k, v in cpu.items()},
        composite_axis=0)

    worst = 0.0
    worst_key = None
    for key, value in trimmed.items():
        other = banded[key]
        if np.isnan(value) and np.isnan(other):
            continue
        scale = max(1.0, abs(value))
        delta = abs(value - other) / scale
        if delta > worst:
            worst, worst_key = delta, key
    print(f"interior cross-check: max relative discrepancy {worst:.3e} "
          f"on {worst_key!r} over {int(interior_mask.sum())} cells")
    assert worst <= INTERIOR_CROSS_CHECK_TOL


def test_attribution_control_specified_zone_only(capsys):
    """Error injected ONLY at d=0 lands on d=0, not on the interior."""
    rng = np.random.default_rng(11)
    ny, nx, nz, width = 26, 32, 3, 5
    gpu = _frame(ny, nx, nz, rng)
    cpu = {k: v.copy() for k, v in gpu.items()}
    bands = dict(sc.strata_masks(ny, nx, width))
    cpu["T2"][bands["d0_specified"]] += 5.0

    spec = sc.compare_masked(gpu, cpu, bands["d0_specified"])
    interior = sc.compare_masked(gpu, cpu, bands["interior"])
    print(f"specified-only injection: d0 T2 MAE {spec['t2_mae']:.6f}, "
          f"interior T2 MAE {interior['t2_mae']:.6f}")
    assert spec["t2_mae"] == pytest.approx(5.0, abs=1e-6)
    assert interior["t2_mae"] == 0.0


def test_attribution_control_interior_only(capsys):
    """The mirror image: interior-only error must not warm the rim."""
    rng = np.random.default_rng(13)
    ny, nx, nz, width = 26, 32, 3, 5
    gpu = _frame(ny, nx, nz, rng)
    cpu = {k: v.copy() for k, v in gpu.items()}
    bands = dict(sc.strata_masks(ny, nx, width))
    cpu["T2"][bands["interior"]] += 3.0

    spec = sc.compare_masked(gpu, cpu, bands["d0_specified"])
    interior = sc.compare_masked(gpu, cpu, bands["interior"])
    print(f"interior-only injection: d0 T2 MAE {spec['t2_mae']:.6f}, "
          f"interior T2 MAE {interior['t2_mae']:.6f}")
    assert interior["t2_mae"] == pytest.approx(3.0, abs=1e-6)
    assert spec["t2_mae"] == 0.0


def test_exclude_rows_path_is_unchanged():
    """The shipped reproduce command's code path keeps its own shape.

    ``compare_pair`` still takes ``(gpu_path, cpu_path, rows)`` and still
    emits exactly the published column set, so the section-7 reproduce
    line runs verbatim.
    """
    import inspect
    assert list(inspect.signature(sc.compare_pair).parameters) == [
        "gpu_path", "cpu_path", "rows"]
    assert sc._COLUMNS == (
        ("domain", "valid_time", "forecast_hour") + sc._METRIC_COLUMNS)
    assert sc._COLUMNS[:3] == ("domain", "valid_time", "forecast_hour")
    # the band table is additive: it carries the same metric columns plus
    # its own two identifiers, and never renames one.
    assert sc._STRATA_COLUMNS[-len(sc._METRIC_COLUMNS):] == sc._METRIC_COLUMNS
    assert "band" in sc._STRATA_COLUMNS and "cells" in sc._STRATA_COLUMNS
