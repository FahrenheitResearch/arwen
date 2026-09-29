"""metgrid's ``search`` walks once per start cell, with every answer unchanged.

A 1 km grid puts some 700 target cells in each 0.25 degree source cell.
Every land target of an island a coarse source holds as sea reaches the
``search`` operator, and each used to walk the same ocean to the same
land on its own: a real island tile (GDAS 0.25 degree, 230 x 230 at 1 km)
spent more than 30 minutes in its root forcing stage doing that.  The
walk depends only on the start cell and the usable mask, so it is taken
once per start cell; these tests hold every answer to the per-target
transcription (``_wps_search_single``) and hold the shared walk to its
cost.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import gpuwm.verify.wps_masked_oracle as horiz


def _reference(field, valid, yy, xx, todo):
    """The per-target transcription, one walk per target cell."""
    out = np.full(yy.shape, np.nan, dtype=np.float64)
    visited = np.zeros(field.shape, dtype=np.int64)
    stamp = 0
    for jj, ii in zip(*np.nonzero(todo)):
        stamp += 1
        out[jj, ii] = horiz._wps_search_single(
            field, valid, float(yy[jj, ii]), float(xx[jj, ii]),
            visited, stamp)
    return out


def _case(rng, ny, nx, density, targets_per_cell):
    field = rng.normal(size=(ny, nx))
    valid = rng.random((ny, nx)) < density
    # Clustered targets: several per start cell, as a fine grid over a
    # coarse source has, plus some off the source grid entirely.
    centres_y = rng.uniform(-1.0, ny, size=12)
    centres_x = rng.uniform(-1.0, nx, size=12)
    yy = (np.repeat(centres_y, targets_per_cell)
          + rng.uniform(-0.6, 0.6, size=12 * targets_per_cell))
    xx = (np.repeat(centres_x, targets_per_cell)
          + rng.uniform(-0.6, 0.6, size=12 * targets_per_cell))
    shape = (12, targets_per_cell)
    todo = rng.random(shape) < 0.9
    return field, valid, yy.reshape(shape), xx.reshape(shape), todo


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("density", [0.002, 0.02, 0.2])
def test_shared_walk_gives_every_target_its_own_walk_s_answer(seed, density):
    rng = np.random.default_rng(seed)
    field, valid, yy, xx, todo = _case(rng, 37, 53, density, 9)
    got = horiz._wps_search(field, valid, yy, xx, todo)
    want = _reference(field, valid, yy, xx, todo)
    np.testing.assert_array_equal(got, want)


def test_shared_walk_keeps_the_queue_limited_answer_and_its_ties():
    """The adversarial queue case, and a symmetric tie, for many targets.

    Donors at (x=0, y=4) and (x=7, y=6): from start (4, 4) the first is
    dequeued while the nearer second is not yet queued, so it wins.  Two
    donors at equal distance from a target: the first found keeps it.
    """
    field = np.zeros((10, 10))
    valid = np.zeros((10, 10), dtype=bool)
    field[4, 0], valid[4, 0] = 11.0, True
    field[6, 7], valid[6, 7] = 22.0, True
    yy = np.array([[4.49, 4.2, 3.9, 4.0]])
    xx = np.array([[4.49, 4.3, 4.1, 3.5]])
    todo = np.ones(yy.shape, dtype=bool)
    got = horiz._wps_search(field, valid, yy, xx, todo)
    np.testing.assert_array_equal(got, _reference(field, valid, yy, xx, todo))
    assert got[0, 0] == 11.0

    tie = np.zeros((9, 9))
    tie_valid = np.zeros((9, 9), dtype=bool)
    tie[4, 2], tie_valid[4, 2] = 1.0, True
    tie[4, 6], tie_valid[4, 6] = 2.0, True
    ty = np.array([[4.0, 4.0]])
    tx = np.array([[4.0, 4.2]])
    todo = np.ones(ty.shape, dtype=bool)
    got = horiz._wps_search(tie, tie_valid, ty, tx, todo)
    np.testing.assert_array_equal(got, _reference(tie, tie_valid, ty, tx, todo))


def test_shared_walk_honours_the_depth_cap(monkeypatch):
    """With the depth counter capped short of the donor, both give NaN."""
    monkeypatch.setattr(horiz, "_WPS_SEARCH_DEPTH", 6)
    rng = np.random.default_rng(3)
    field = rng.normal(size=(40, 40))
    valid = np.zeros((40, 40), dtype=bool)
    valid[2, 2] = True
    valid[20, 35] = True
    yy = np.array([[20.2, 20.4, 3.0, 5.1]])
    xx = np.array([[20.1, 20.3, 3.2, 2.4]])
    todo = np.ones(yy.shape, dtype=bool)
    got = horiz._wps_search(field, valid, yy, xx, todo)
    want = _reference(field, valid, yy, xx, todo)
    np.testing.assert_array_equal(got, want)
    assert np.isnan(got[0, 0]) and np.isfinite(got[0, 2])


def test_an_island_a_coarse_source_holds_as_sea_walks_once_per_source_cell(
        monkeypatch):
    """The island tile's shape: a whole-globe 0.25 degree source whose only
    usable land is 60 cells away, and 2,100 fine-grid island targets in
    three source cells.  One walk per target took ~2,100 walks of some
    7,000 source cells each (tens of seconds); one per start cell is three.
    """
    ny, nx = 721, 1440
    field = np.full((ny, nx), 290.0)
    valid = np.zeros((ny, nx), dtype=bool)
    valid[400:404, 700:704] = True
    field[400:404, 700:704] = 300.0 + np.arange(16).reshape(4, 4)
    rng = np.random.default_rng(0)
    centres_x = np.array([760.0, 761.0, 761.0])
    centres_y = np.array([402.0, 402.0, 403.0])
    yy = (np.repeat(centres_y, 700)
          + rng.uniform(-0.49, 0.49, size=2100)).reshape(3, 700)
    xx = (np.repeat(centres_x, 700)
          + rng.uniform(-0.49, 0.49, size=2100)).reshape(3, 700)
    todo = np.ones(yy.shape, dtype=bool)

    walks = []
    real = getattr(horiz, "_wps_search_candidates", None)
    assert real is not None, "the search walks once per target cell"

    def counted(*args):
        walks.append(args[-1])
        return real(*args)

    monkeypatch.setattr(horiz, "_wps_search_candidates", counted)
    started = time.perf_counter()
    got = horiz._wps_search(field, valid, yy, xx, todo)
    seconds = time.perf_counter() - started
    assert len(walks) == 3
    assert seconds < 5.0
    assert np.all((got >= 300.0) & (got <= 315.0))
    # Spot-check the answers against one walk per target.
    pick = (slice(None), slice(0, 700, 70))
    np.testing.assert_array_equal(
        got[pick], _reference(field, valid, yy[pick], xx[pick],
                              todo[pick]))
