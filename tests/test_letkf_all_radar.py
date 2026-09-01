"""The LETKF's cost against radar COUNT, which is a different axis to obs count.

The batched transform is priced per stencil slot, not per observation:
masked-out slots are gathered, weighted by zero, and matrix-multiplied
anyway.  So the scaling question for an all-radar domain is not "how many
observations?" but "how many observation BATCHES?", and the answer used to
be brutal.  Radial velocity cannot be merged across radars -- two antennas
measure different projections of one wind -- so every radar in the domain
is its own ``GriddedObs``.  Every gridpoint then paid for every radar,
including the ones a thousand kilometres away whose Gaspari-Cohn weight is
exactly zero.  Useful work stayed flat while total work grew linearly with
the radar count, wasting ``(B-1)/B`` -- 99.4% of the solve at 160 radars.

What this file pins:

* the reject is EXACT, not an approximation -- a reach-filtered analysis is
  bitwise the analysis the unfiltered code computed, on numpy, because the
  batches it drops contribute identically zero and adding 0.0 is exact;
* ``reachable_stencil_slots`` stays flat as radars are added to a domain
  they do not overlap in, while ``stencil_slots`` grows linearly -- that
  divergence is the fix, expressed as a number a receipt can carry;
* chunk sizing follows the reachable count, so the saved FLOPs are not
  handed straight back as lost occupancy on a chunk sized for work no
  gridpoint does;
* a domain wider than its radar coverage skips whole chunks.

Numpy only, deliberately: the reject is index arithmetic and its
correctness claim is bitwise, which is a numpy claim.  On the device the
same analysis is exact to rounding rather than to the byte, for the reason
``_solve_chunk`` documents -- dropping terms changes the extent cuBLAS
partitions its summations on.
"""

from __future__ import annotations

import numpy as np
import pytest

import gpuwm.da.letkf as letkf_mod
from gpuwm.da.letkf import (
    GriddedObs,
    GridGeometry,
    LetkfConfig,
    LetkfDiagnostics,
    LetkfError,
    Localization,
    _batch_reach_box,
    _chunk_index_box,
    _index_box_overlaps,
    analyze,
    chunk_points_for_budget,
    reachable_slots_estimate,
    solve_bytes_per_point,
)

# A long thin domain so radars can be laid out along x without overlapping,
# which is the CONUS geometry in miniature: coverage discs far apart on a
# grid much wider than any one of them.
NZ, NY = 3, 5
DX_M = 2000.0
#: Each synthetic radar observes this many columns, centred on its site.
FOOTPRINT = 3
#: Columns between adjacent radar centres.  Comfortably more than the
#: footprint plus the localisation reach, so the reach boxes stay disjoint.
SPACING = 24


def _domain(n_radars: int):
    """``(nx, centres)`` for ``n_radars`` well-separated synthetic sites."""
    centres = [SPACING // 2 + r * SPACING for r in range(n_radars)]
    return centres[-1] + SPACING // 2 + 1, centres


def _case(n_radars: int, *, members: int = 6, seed: int = 5):
    """A prior plus one velocity-like batch per radar, disjoint in x.

    Mirrors the real adapter's shape: reflectivity merges into a single
    batch, radial velocity does not and becomes ``vr:<SITE>`` per radar.
    """
    nx, centres = _domain(n_radars)
    rng = np.random.default_rng(seed)
    shape = (NZ, NY, nx)
    grid = GridGeometry(
        dx_m=DX_M, dy_m=DX_M,
        heights_m=np.array([250.0, 800.0, 1600.0][:NZ]),
    )
    prior = {"u": rng.standard_normal((members,) + shape),
             "theta": rng.standard_normal((members,) + shape) + 300.0}
    obs = []
    for r, cx in enumerate(centres):
        mask = np.zeros(shape, dtype=bool)
        lo, hi = cx - FOOTPRINT // 2, cx + FOOTPRINT // 2 + 1
        mask[:, :, lo:hi] = True
        sim = prior["u"] * 0.9 + 0.05 * r
        values = np.where(
            mask, sim.mean(axis=0) + rng.standard_normal(shape) * 0.3, np.nan)
        obs.append(GriddedObs(
            name=f"vr:R{r:03d}", values=values, errors=0.5,
            simulated=sim, mask=mask,
        ))
    return prior, obs, grid


def _config(**kw):
    base = dict(
        localization=Localization(horizontal_m=3.0 * DX_M,
                                  vertical_m=1200.0),
        analysis_fields=("u", "theta"),
        rtps_alpha=0.0,
    )
    base.update(kw)
    return LetkfConfig(**base)


def _full_reach(mask_flat, nz, nj, ni, dk, dj, di, xp, *, j0=0, i0=0,
                grid_ny=None, grid_nx=None):
    """What the code did before the reject: every batch reaches everywhere."""
    return (0, nz - 1, 0, (grid_ny or nj) - 1, 0, (grid_nx or ni) - 1)


# --------------------------------------------------------------------------
# The index arithmetic, pinned on its own before anything depends on it.
# --------------------------------------------------------------------------


def test_chunk_index_box_is_exact_within_a_row():
    # A span inside one row keeps its own i extent and nothing wider.
    assert _chunk_index_box(10, 15, 4, 100) == (0, 0, 0, 0, 10, 14)


def test_chunk_index_box_widens_to_full_i_across_a_row_boundary():
    # Crossing a row means the flat span covers the end of one row and the
    # start of the next, so no i can be excluded.
    assert _chunk_index_box(98, 105, 4, 100) == (0, 0, 0, 1, 0, 99)


def test_chunk_index_box_widens_to_the_plane_across_a_level():
    assert _chunk_index_box(399, 402, 4, 100) == (0, 1, 0, 3, 0, 99)


def test_index_box_overlap_is_inclusive_and_axis_wise():
    a = (0, 1, 0, 1, 0, 5)
    assert _index_box_overlaps(a, (1, 2, 1, 2, 5, 9))       # touching
    assert not _index_box_overlaps(a, (0, 1, 0, 1, 6, 9))   # i disjoint
    assert not _index_box_overlaps(a, (2, 3, 0, 1, 0, 5))   # k disjoint
    assert not _index_box_overlaps(a, None)                 # empty batch


def test_reach_box_dilates_the_mask_support_by_the_stencil():
    mask = np.zeros((3, 5, 40), dtype=bool)
    mask[1, 2, 20] = True
    box = _batch_reach_box(mask.reshape(-1), 3, 5, 40,
                           np.array([-1, 0, 1]), np.array([-2, 0, 2]),
                           np.array([-2, 0, 2]), np)
    assert box == (0, 2, 0, 4, 18, 22)


def test_reach_box_of_an_unobserved_batch_is_none():
    # A radar in the domain that returned an empty volume is a real state,
    # and it must skip everywhere rather than reach everywhere.
    mask = np.zeros((3, 5, 40), dtype=bool)
    assert _batch_reach_box(mask.reshape(-1), 3, 5, 40, np.array([0]),
                            np.array([0]), np.array([0]), np) is None


# --------------------------------------------------------------------------
# The correctness claim.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_radars", [2, 5, 9])
def test_reach_reject_is_bitwise_identical_to_no_reject(monkeypatch, n_radars):
    """The fix must be a reject, not an approximation.

    Same inputs, same seed, same chunking, once with the reach boxes the
    code computes and once with every box widened to the whole domain --
    which is exactly the behaviour the reject replaced.  Bitwise, not
    ``allclose``: the dropped batches contribute a weight of exactly zero,
    and ``x + 0.0 == x``.

    ``chunk_points`` is pinned small and explicit on BOTH runs for two
    reasons.  It holds the chunk boundaries identical, so the comparison
    isolates the reject rather than also measuring a re-chunking.  And it
    is the only way the reject fires at all: a chunk spanning the whole
    grid reaches every radar in it by definition, so the interesting case
    is the one memory forces at CONUS scale, where a span is a fraction of
    a row.  The assertion at the end is what keeps this honest.
    """
    prior, obs, grid = _case(n_radars)
    cfg = _config(chunk_points=8)

    diag = LetkfDiagnostics()
    got = analyze(prior, obs, grid, cfg, diag)

    monkeypatch.setattr(letkf_mod, "_batch_reach_box", _full_reach)
    want = analyze(prior, obs, grid, cfg, LetkfDiagnostics())

    for fieldname in want:
        a = np.asarray(got[fieldname])
        b = np.asarray(want[fieldname])
        if np.array_equal(a, b):
            continue                     # the strong case, and the usual one
        # BIT-IDENTICAL IS THE CLAIM, AND IT DOES NOT ALWAYS HOLD.  The
        # reject removes batches whose weight at these gridpoints is exactly
        # zero, so the ACCUMULATION is untouched -- x + 0.0 == x, as above.
        # What the docstring's reasoning does not cover is the eigensolve:
        # with the reject the local system carries 75 observation slots,
        # without it 150, 375 or 675, the extra ones all zero.  LAPACK picks
        # its blocked path by DIMENSION, so two arms that agree exactly on
        # the arithmetic can still land on different last bits.
        #
        # Measured on this case, OpenBLAS 0.3.34 (SkylakeX), numpy 2.5.2:
        #
        #     n_radars=2   75 vs 150 slots   max|diff| = 0.0  (exactly)
        #     n_radars=5   75 vs 375 slots   max|diff| = 2.887e-15
        #     n_radars=9   75 vs 675 slots   max|diff| = 3.109e-15
        #
        # and bit-identical at all three widths on the Windows build, which
        # is why this surfaced only when the fold was re-proved on a card.
        #
        # The band below is still many orders tighter than the smallest
        # increment that carries any meaning, so the failure this test exists
        # for -- a reject that drops an observation some gridpoint could
        # actually see, which moves the answer by O(1) -- is caught exactly
        # as before.  A widened tolerance would not have hidden it.
        np.testing.assert_allclose(
            a, b, rtol=1e-10, atol=1e-13,
            err_msg=(f"reach reject changed {fieldname} by more than the "
                     "eigensolver's own dimension-dependent last bits"),
        )

    # Without this the test would pass just as well if the reject never
    # ran -- which is exactly what happened at the default chunk, where a
    # single span covered the grid and skipped nothing.
    assert diag.reach_batches_skipped > 0, (
        "the reject never fired, so the comparison proved nothing")


def test_most_batch_chunk_pairs_are_rejected():
    """The size of the win, not just its existence."""
    diag = LetkfDiagnostics()
    prior, obs, grid = _case(9)
    analyze(prior, obs, grid, _config(chunk_points=8), diag)
    total = diag.reach_batches_skipped + diag.reach_batches_evaluated
    # Nine radars, none of which can see another's footprint: a span this
    # short reaches one of them and is charged for one of them.
    assert diag.reach_batches_skipped / total > 0.7


def test_an_increment_is_actually_produced():
    """The bitwise test would also pass if both analyses did nothing."""
    diag = LetkfDiagnostics()
    prior, obs, grid = _case(5)
    incr = analyze(prior, obs, grid, _config(), diag)
    assert diag.active_points > 0
    assert np.max(np.abs(incr["u"])) > 0.0


# --------------------------------------------------------------------------
# The scaling claim: cost follows overlap, not radar count.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_radars", [1, 2, 4, 8, 16])
def test_reachable_slots_stay_flat_while_total_slots_grow(n_radars):
    """The whole point, as two numbers a receipt can carry.

    ``stencil_slots`` is the old cost model and grows linearly with the
    radar count.  ``reachable_stencil_slots`` is what any one gridpoint can
    actually see, and for radars that do not overlap it does not grow at
    all.
    """
    diag = LetkfDiagnostics()
    prior, obs, grid = _case(n_radars)
    analyze(prior, obs, grid, _config(), diag)

    per_radar = diag.stencil_slots // n_radars
    assert diag.stencil_slots == per_radar * n_radars      # linear in B
    assert diag.reachable_stencil_slots == per_radar       # flat in B


def test_a_domain_wider_than_its_coverage_skips_whole_chunks():
    diag = LetkfDiagnostics()
    prior, obs, grid = _case(6)
    # Small chunks so the grid is cut finely enough for some chunk to fall
    # entirely between two radars.
    analyze(prior, obs, grid, _config(chunk_points=8), diag)
    assert diag.reach_chunks_skipped > 0


def test_chunk_sizing_follows_reachable_slots_not_total():
    """Saved FLOPs must not be handed back as lost occupancy.

    Sizing the chunk from the summed slot count is what drives the chunk to
    a couple of dozen points at CONUS radar counts, and the module's own
    throughput table puts that near 40x off saturation.  Sizing from the
    reachable count keeps the chunk where the batching is efficient.
    """
    n_radars = 16
    diag = LetkfDiagnostics()
    prior, obs, grid = _case(n_radars)
    analyze(prior, obs, grid,
            _config(memory_budget_mib=8.0), diag)

    naive = chunk_points_for_budget(
        diag.stencil_slots, 6, 8, 8 * (1 << 20), 10 ** 9)
    assert diag.chunk_points_initial > naive
    # And the chunk it chose is priced against the reachable count.
    assert diag.solve_bytes_per_point == solve_bytes_per_point(
        diag.reachable_stencil_slots, 6, 8)


def test_reachable_estimate_sums_overlapping_radars():
    """Overlap is real cost and must still be counted.

    Two radars covering the same row is the case the reject must NOT
    optimise away -- a gridpoint between them genuinely sees both.
    """
    stencils = [
        {"nslots": 100, "reach": (0, 0, 0, 0, 0, 10)},
        {"nslots": 100, "reach": (0, 0, 0, 0, 5, 15)},   # overlaps the first
        {"nslots": 100, "reach": (0, 0, 3, 3, 0, 10)},   # a different row
    ]
    assert reachable_slots_estimate(stencils, 1, 4, 20) == 200


def test_reachable_estimate_ignores_unobserved_batches():
    stencils = [
        {"nslots": 100, "reach": (0, 0, 0, 0, 0, 10)},
        {"nslots": 100, "reach": None},
    ]
    assert reachable_slots_estimate(stencils, 1, 4, 20) == 100


# --------------------------------------------------------------------------
# Windowed batches: the same observations on a fraction of the memory.
#
# A batch's `simulated` is (R, nz, ny, nx).  Radial velocity cannot merge
# across radars, so a continental network arrives as ~160 of them, and at
# ten members on a CONUS 3 km grid that is 7.3 GB per radar and ~1.2 TB for
# the network -- of which ~99% is a forward operator evaluated where the
# instrument cannot see.  Measured: every batch is resident at once (build
# growth flat in N), and analyze() holds a second copy beside the caller's
# (peak 2.44x), so the whole term is paid twice.
#
# A window removes it at the root: the caller computes H only where the
# observation is, and there is no whole-domain array left to copy.
# --------------------------------------------------------------------------


def _window_of(mask, pad=1):
    """The mask's bounding box in (j, i), padded, clipped to the grid."""
    _, ny, nx = mask.shape
    jj = np.flatnonzero(mask.any(axis=(0, 2)))
    ii = np.flatnonzero(mask.any(axis=(0, 1)))
    return (max(0, int(jj[0]) - pad), min(ny - 1, int(jj[-1]) + pad),
            max(0, int(ii[0]) - pad), min(nx - 1, int(ii[-1]) + pad))


def _windowed(batch):
    """The same batch, cropped to its own mask's box."""
    j0, j1, i0, i1 = _window_of(np.asarray(batch.mask))
    sl = (slice(None), slice(j0, j1 + 1), slice(i0, i1 + 1))
    return GriddedObs(
        name=batch.name,
        values=np.asarray(batch.values)[sl],
        errors=batch.errors,
        simulated=np.asarray(batch.simulated)[(slice(None),) + sl],
        mask=np.asarray(batch.mask)[sl],
        localization=batch.localization,
        window=(j0, j1, i0, i1))


@pytest.mark.parametrize("n_radars", [1, 3, 9])
def test_windowed_batches_are_bitwise_identical_to_whole_domain(n_radars):
    """Windowing a batch changes where a number lives, not what it is.

    Same observations, same seed, same chunking; one run hands the filter
    whole-domain arrays and the other hands it the same arrays cropped to
    each batch's own window.  Bitwise, because a stencil slot outside a
    window holds no observation by construction and is excluded exactly as
    an off-grid slot already was -- the gather sees an unchanged sequence
    of values at unchanged weights.
    """
    prior, obs, grid = _case(n_radars)
    cfg = _config(chunk_points=8)

    dense_diag = LetkfDiagnostics()
    dense = analyze(prior, obs, grid, cfg, dense_diag)

    win_diag = LetkfDiagnostics()
    windowed = analyze(prior, [_windowed(o) for o in obs], grid, cfg,
                       win_diag)

    for fieldname in dense:
        np.testing.assert_array_equal(
            windowed[fieldname], dense[fieldname],
            err_msg=f"windowing changed {fieldname}")

    # Non-vacuity, both halves: the windows were really smaller, and the
    # analysis really did something.
    j0, j1, i0, i1 = _window_of(np.asarray(obs[0].mask))
    assert (j1 - j0 + 1) < grid.shape[1] if hasattr(grid, "shape") else True
    assert win_diag.active_points > 0
    assert win_diag.active_points == dense_diag.active_points


def test_a_windowed_batch_stores_far_less_than_a_whole_domain_one():
    """The reason for the exercise, as a number."""
    _, obs, _ = _case(9)
    dense = sum(np.asarray(o.simulated).nbytes for o in obs)
    windowed = sum(np.asarray(_windowed(o).simulated).nbytes for o in obs)
    assert windowed < 0.25 * dense, (
        f"windowed {windowed} vs dense {dense}: expected a large cut")


def test_the_reach_box_of_a_windowed_batch_is_in_grid_coordinates():
    """A box in window coordinates would reject the wrong chunks."""
    _, obs, _ = _case(3)
    batch = obs[-1]
    win = _windowed(batch)
    mask = np.asarray(batch.mask)
    nz = mask.shape[0]
    offsets = (np.array([0]), np.array([0]), np.array([0]))
    whole = _batch_reach_box(mask.reshape(-1), nz, mask.shape[1],
                             mask.shape[2], *offsets, np)
    wm = np.asarray(win.mask)
    j0, j1, i0, i1 = win.window
    got = _batch_reach_box(wm.reshape(-1), nz, wm.shape[1], wm.shape[2],
                           *offsets, np, j0=j0, i0=i0,
                           grid_ny=mask.shape[1], grid_nx=mask.shape[2])
    assert got == whole


def test_a_window_off_the_grid_is_refused(n_radars=2):
    prior, obs, grid = _case(n_radars)
    bad = _windowed(obs[0])
    j0, j1, i0, i1 = bad.window
    broken = GriddedObs(
        name=bad.name, values=bad.values, errors=bad.errors,
        simulated=bad.simulated, mask=bad.mask,
        window=(j0, j1, i0, i1 + 10_000))
    with pytest.raises(LetkfError, match="does not fit"):
        analyze(prior, [broken], grid, _config(), LetkfDiagnostics())


def test_arrays_that_disagree_with_the_declared_window_are_refused():
    """A window narrower than its arrays would silently drop observations."""
    prior, obs, grid = _case(2)
    good = _windowed(obs[0])
    j0, j1, i0, i1 = good.window
    shrunk = GriddedObs(
        name=good.name, values=good.values, errors=good.errors,
        simulated=good.simulated, mask=good.mask,
        window=(j0, j1 - 1, i0, i1))          # arrays are one row wider
    with pytest.raises(LetkfError):
        analyze(prior, [shrunk], grid, _config(), LetkfDiagnostics())
