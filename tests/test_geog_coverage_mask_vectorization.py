"""Differential fuzz: vectorized tile coverage vs the frozen scalar oracle.

``GeogDataset.tile_coverage_mask`` used to probe the tile inventory once per
source CELL through a per-row ``np.fromiter`` over a generator.  Tile presence
depends only on the ``(x_origin, y_origin)`` pair, so a domain window spanning
a handful of distinct origins was paying millions of dict lookups to answer a
handful of questions.  ``GeogDataset.missing_tiles`` had the matching problem
on its own side: a Python loop over every missing cell.

Both are now vectorized.  The functions below are VERBATIM COPIES of the
pre-vectorization implementations, frozen here as reference oracles.  Do not
"tidy" them to share code with the production versions -- their whole value is
that they are an independent second implementation.  The fuzz comparison is
bitwise (``np.array_equal`` on the boolean masks, tuple equality on the
origins) across synthetic tile inventories that exercise dense-global,
sparse-missing-tile, staged-half and non-wrapping-regional trees, including
windows with negative ``x0``, ``y0 < 1``, ``y1 > ny_global`` and wrapped x
ranges.

The existing coverage in tests/test_static_build.py and
tests/test_ingest_preflight.py only ever looks at 4x4 windows, which cannot
distinguish the two implementations.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.static.geog import GeogDataset


# ---------------------------------------------------------------------------
# Frozen reference oracles (the pre-vectorization implementations, verbatim)
# ---------------------------------------------------------------------------

def reference_tile_coverage_mask(self: GeogDataset, x0: int, x1: int,
                                 y0: int, y1: int) -> np.ndarray:
    """Pre-vectorization ``GeogDataset.tile_coverage_mask``, unchanged."""

    if x1 < x0 or y1 < y0:
        raise ValueError("empty window")
    x = np.arange(x0, x1 + 1, dtype=np.int64)
    y = np.arange(y0, y1 + 1, dtype=np.int64)
    if self.wraps_x:
        source_x = (x - 1) % self.nx_global + 1
        x_inside = np.ones(x.size, dtype=bool)
    else:
        source_x = x
        x_inside = (x >= 1) & (x <= self.nx_global)
    y_inside = (y >= 1) & (y <= self.ny_global)
    x_origins = ((source_x - 1) // self.index.tile_x
                 * self.index.tile_x + 1)
    y_origins = ((y - 1) // self.index.tile_y
                 * self.index.tile_y + 1)
    mask = np.zeros((y.size, x.size), dtype=bool)
    for j, (ys, inside_y) in enumerate(zip(y_origins, y_inside)):
        if not inside_y:
            continue
        mask[j] = x_inside & np.fromiter(
            ((int(xs), int(ys)) in self.tiles for xs in x_origins),
            dtype=bool, count=x.size,
        )
    return mask


def reference_missing_tiles(self: GeogDataset, x0: int, x1: int, y0: int,
                            y1: int) -> tuple[tuple[int, int], ...]:
    """Pre-vectorization ``GeogDataset.missing_tiles``, unchanged."""

    mask = reference_tile_coverage_mask(self, x0, x1, y0, y1)
    missing_cells = ~mask & self._extent_mask(x0, x1, y0, y1)
    missing: set[tuple[int, int]] = set()
    for j, i in np.argwhere(missing_cells):
        x = x0 + int(i)
        y = y0 + int(j)
        if self.wraps_x:
            x = (x - 1) % self.nx_global + 1
        xs = (x - 1) // self.index.tile_x * self.index.tile_x + 1
        ys = (y - 1) // self.index.tile_y * self.index.tile_y + 1
        missing.add((xs, ys))
    return tuple(sorted(missing, key=lambda item: (item[1], item[0])))


# ---------------------------------------------------------------------------
# Synthetic tile inventories
# ---------------------------------------------------------------------------

def _write_index(dirpath, **over) -> dict[str, object]:
    kv: dict[str, object] = dict(
        type="continuous", signed="yes", projection="regular_ll",
        dx=1.0, dy=1.0, known_x=1.0, known_y=1.0,
        known_lat=-89.5, known_lon=-179.5, wordsize=2,
        tile_x=30, tile_y=30, tile_z=1,
    )
    kv.update(over)
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "index").write_text(
        "\n".join(f"{k} = {v}" for k, v in kv.items()) + "\n",
        encoding="utf-8")
    return kv


def _write_tiles(dirpath, kv, *, nxg: int, nyg: int, keep=None) -> None:
    """Write zero-filled tiles.  Coverage is metadata-only: no payload read."""

    tx, ty = int(kv["tile_x"]), int(kv["tile_y"])
    words = int(kv["tile_z"]) * ty * tx
    payload = np.zeros(words, dtype=">i2")
    for ys in range(1, nyg + 1, ty):
        for xs in range(1, nxg + 1, tx):
            if keep is not None and not keep(xs, ys):
                continue
            payload.tofile(
                dirpath / f"{xs:05d}-{xs + tx - 1:05d}."
                          f"{ys:05d}-{ys + ty - 1:05d}")


def _dense_global(root):
    """360x180 one-degree global tree, every tile present, wraps in x."""

    path = root / "dense_global"
    kv = _write_index(path, dx=1.0, dy=1.0)
    _write_tiles(path, kv, nxg=360, nyg=180)
    return GeogDataset(path)


def _sparse_missing(root):
    """Same geometry with a deterministic scatter of absent tiles."""

    path = root / "sparse_missing"
    kv = _write_index(path, dx=1.0, dy=1.0)
    absent = {(31, 31), (61, 1), (61, 121), (121, 61), (181, 91),
              (331, 151), (1, 1), (301, 31)}
    _write_tiles(path, kv, nxg=360, nyg=180,
                 keep=lambda xs, ys: (xs, ys) not in absent)
    return GeogDataset(path, sparse=True)


def _staged_half(root):
    """Footprint-minimized staging: only the eastern half is on disk."""

    path = root / "staged_half"
    kv = _write_index(path, dx=1.0, dy=1.0)
    _write_tiles(path, kv, nxg=360, nyg=180, keep=lambda xs, ys: xs >= 181)
    return GeogDataset(path, sparse=True)


# A spacing that does not divide 360/180 evenly leaves the regular-LL global
# geometry un-inferable, which is what keeps a regional tree non-wrapping even
# when it declares itself sparse (GeogDataset treats `sparse` as proof of a
# footprint-minimized staging tree and would otherwise restore global extent).
_REGIONAL = dict(dx=0.7, dy=0.7, tile_x=20, tile_y=20,
                 known_lat=25.0, known_lon=-110.0)


def _regional(root):
    """Non-wrapping regional tree: 100x60 cells, no wrap."""

    path = root / "regional"
    kv = _write_index(path, **_REGIONAL)
    _write_tiles(path, kv, nxg=100, nyg=60)
    return GeogDataset(path)


def _sparse_regional(root):
    """Non-wrapping regional tree with holes: exercises both edge classes."""

    path = root / "sparse_regional"
    kv = _write_index(path, **_REGIONAL)
    absent = {(21, 21), (81, 1), (41, 41)}
    _write_tiles(path, kv, nxg=100, nyg=60,
                 keep=lambda xs, ys: (xs, ys) not in absent)
    return GeogDataset(path, sparse=True)


_BUILDERS = (
    ("dense-global", _dense_global, True),
    ("sparse-missing-tiles", _sparse_missing, True),
    ("staged-half", _staged_half, True),
    ("non-wrapping-regional", _regional, False),
    ("sparse-non-wrapping-regional", _sparse_regional, False),
)


@pytest.fixture(scope="module")
def inventories(tmp_path_factory):
    root = tmp_path_factory.mktemp("geog_fuzz")
    built = {}
    for name, builder, wraps in _BUILDERS:
        ds = builder(root)
        assert ds.wraps_x is wraps, (
            f"{name} synthetic tree did not get the intended wrap geometry")
        built[name] = ds
    return built


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------

def _edge_windows(ds: GeogDataset) -> list[tuple[int, int, int, int]]:
    """Hand-picked boundary cases the random draw should not be trusted to hit."""

    nx, ny = ds.nx_global, ds.ny_global
    tx, ty = ds.index.tile_x, ds.index.tile_y
    windows = [
        (1, 1, 1, 1),                       # single cell, origin
        (1, nx, 1, ny),                     # the entire globe/region
        (-20, 5, 1, 10),                    # negative x0
        (-1, 0, 1, 4),                      # wholly negative/zero x
        (1, 6, 0, 4),                       # y0 < 1
        (1, 6, -7, 3),                      # y0 far below the extent
        (1, 6, ny - 2, ny + 9),             # y1 > ny_global
        (1, 6, ny + 3, ny + 8),             # wholly above the extent
        (1, 6, -4, ny + 4),                 # straddles both y edges
        (nx - 3, nx + 6, 1, 6),             # crosses the x seam
        (nx - 1, nx + 1, ny - 1, ny + 1),   # far corner, both edges
        (tx - 1, tx + 2, ty - 1, ty + 2),   # straddles one tile boundary
        (tx, tx + 1, ty, ty + 1),           # last/first cell of adjacent tiles
        (1, nx, -3, ny + 3),                # full width, over-tall
        (2 * nx + 1, 2 * nx + 9, 1, 5),     # x far past one full wrap
        (-nx - 5, -nx + 4, 2, 7),           # x far below one full wrap
        (-5, nx + 5, 1, 4),                 # wider than the axis, both ends out
        (-5, nx + 5, -2, ny + 2),           # over-wide and over-tall
    ]
    return [w for w in windows if w[1] >= w[0] and w[3] >= w[2]]


def _random_windows(rng, ds: GeogDataset, count: int):
    nx, ny = ds.nx_global, ds.ny_global
    out = []
    for _ in range(count):
        width = int(rng.integers(1, min(nx, 120) + 1))
        height = int(rng.integers(1, min(ny, 60) + 1))
        x0 = int(rng.integers(-nx - 10, nx + 11))
        y0 = int(rng.integers(-10, ny + 11))
        out.append((x0, x0 + width - 1, y0, y0 + height - 1))
    return out


# ---------------------------------------------------------------------------
# The differential fuzz
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [spec[0] for spec in _BUILDERS])
def test_tile_coverage_mask_matches_scalar_oracle_bitwise(name, inventories):
    ds = inventories[name]
    rng = np.random.default_rng(20260807)
    windows = _edge_windows(ds) + _random_windows(rng, ds, 200)

    checked = 0
    for x0, x1, y0, y1 in windows:
        expected = reference_tile_coverage_mask(ds, x0, x1, y0, y1)
        actual = ds.tile_coverage_mask(x0, x1, y0, y1)
        assert actual.shape == expected.shape, (name, x0, x1, y0, y1)
        assert actual.dtype == expected.dtype == np.dtype(bool)
        assert np.array_equal(actual, expected), (
            f"{name}: coverage mask differs for window "
            f"x={x0}..{x1}, y={y0}..{y1}")
        checked += 1

    assert checked >= 200


@pytest.mark.parametrize("name", [spec[0] for spec in _BUILDERS])
def test_missing_tiles_matches_scalar_oracle(name, inventories):
    ds = inventories[name]
    rng = np.random.default_rng(20260807)
    windows = _edge_windows(ds) + _random_windows(rng, ds, 200)

    saw_missing = False
    for x0, x1, y0, y1 in windows:
        expected = reference_missing_tiles(ds, x0, x1, y0, y1)
        actual = ds.missing_tiles(x0, x1, y0, y1)
        assert actual == expected, (
            f"{name}: missing_tiles differs for window "
            f"x={x0}..{x1}, y={y0}..{y1}")
        assert all(isinstance(v, int) for pair in actual for v in pair)
        saw_missing |= bool(actual)

    # Only the trees with deliberate holes can report a missing origin; a
    # fully-populated tree correctly reports none however the window is placed.
    if name in ("sparse-missing-tiles", "staged-half",
                "sparse-non-wrapping-regional"):
        assert saw_missing, f"{name} never produced a missing origin"
    else:
        assert not saw_missing, f"{name} is fully populated but reported holes"


def test_fuzz_windows_actually_exercise_the_edges(inventories):
    """The instrument, tested: the draw really does hit every edge class."""

    ds = inventories["dense-global"]
    rng = np.random.default_rng(20260807)
    windows = _edge_windows(ds) + _random_windows(rng, ds, 200)

    assert any(x0 < 0 for x0, _, _, _ in windows)
    assert any(y0 < 1 for _, _, y0, _ in windows)
    assert any(y1 > ds.ny_global for _, _, _, y1 in windows)
    assert any(x1 > ds.nx_global for _, x1, _, _ in windows)
    assert any(x0 < 1 and x1 > ds.nx_global for x0, x1, _, _ in windows)
    # partially-covered (not all-true, not all-false) masks must occur, or the
    # comparison would be blind to origin-expansion bugs
    partial = 0
    for x0, x1, y0, y1 in windows:
        mask = ds.tile_coverage_mask(x0, x1, y0, y1)
        if mask.any() and not mask.all():
            partial += 1
    assert partial >= 5


def test_oracle_detects_a_perturbed_mask(inventories):
    """Negative control: the comparator must fail on a one-bit difference."""

    ds = inventories["sparse-missing-tiles"]
    window = (1, 90, 1, 90)
    expected = reference_tile_coverage_mask(ds, *window)
    tampered = expected.copy()
    tampered[0, 0] = not tampered[0, 0]
    assert not np.array_equal(tampered, expected)
    assert np.array_equal(ds.tile_coverage_mask(*window), expected)
