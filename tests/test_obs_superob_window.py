"""Windowed superob accumulators: the same observations, on 1% of the memory.

A radar is a local instrument on a domain that need not be.  Its gates
occupy a disc about 250 km across; a continental analysis grid is some
5000 km across.  Full-domain accumulators therefore spent 99% of their
bytes holding zeros the antenna could never write to -- measured at 120
bytes per cell per site, which is 10.9 GB per radar on a CONUS 3 km grid
and the single thing standing between a regional system and a continental
one.

The fix windows each contribution to the cells its own radar can reach.
The correctness claim is the strong one, and this file is where it is
made: windowing is a change of ADDRESS, not of arithmetic.  Every value
accumulated is the same float, applied in the same order, to the same
logical cell -- ``np.add.at`` sees an unchanged sequence of additions --
so a windowed analysis is bitwise the analysis the dense code produced,
not merely close to it.

What this file pins:

* bitwise equality against the dense path, on a multi-radar fixture, for
  every array a merge produces -- and an assertion that the window was
  actually smaller than the domain, so the comparison is not vacuous;
* no observation is lost: the windowed contribution accumulates exactly
  the gate counts the dense one did;
* the guard -- a window that under-covers its radar raises rather than
  silently dropping gates, which is the one failure an analysis could not
  detect downstream;
* the degenerate cases: a radar off the grid, and a merge handed a window
  that does not fit.
"""

from __future__ import annotations

import numpy as np
import pytest

import gpuwm.obs.superob as superob_mod
from gpuwm.obs.superob import (
    RadarContribution,
    SuperobError,
    SuperobParams,
    horizontal_window,
    merge_contributions,
    superob_volume,
)
from gpuwm.obs.sweeps import Moment, RadarSite, RadarVolume, Sweep
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.lambert import LambertGrid

REF_LAT, REF_LON = 35.3331, -97.2778
#: A domain several times wider than the range authority below, which is
#: the whole point: on a grid smaller than one radar's disc every window
#: is the domain and windowing cannot be observed.
NX = NY = 100
DX_M = 2000.0
NZ = 8
#: Shorter than the operational 250 km so the fixture stays fast while the
#: window still covers only a fraction of the domain.
RANGE_KM = 40.0


def _grid() -> TargetGrid:
    projection = LambertGrid(
        ref_lat=REF_LAT, ref_lon=REF_LON, truelat1=33.0, truelat2=37.0,
        stand_lon=REF_LON, dx=DX_M, dy=DX_M, e_we=NX + 1, e_sn=NY + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, 8000.0, NZ + 1), name="analytic")


def _params() -> SuperobParams:
    return SuperobParams(max_range_km=RANGE_KM).validate()


def _volume_at(grid, *, site_id: str, j: int, i: int,
               gates: int = 150) -> RadarVolume:
    """A synthetic volume whose antenna sits at grid index ``(j, i)``.

    Enough gates to reach the range authority, so the accumulators are
    filled out to the window's edge rather than rattling around inside it.
    """
    azimuth = np.arange(0.0, 360.0, 6.0, dtype=np.float32)
    ref = np.tile(np.linspace(15.0, 55.0, gates)[None, :],
                  (azimuth.size, 1)).astype(np.float32)
    vel = np.tile(np.linspace(-18.0, 18.0, gates)[None, :],
                  (azimuth.size, 1)).astype(np.float32)
    sweeps = []
    for index, elevation in enumerate((0.5, 1.5)):
        sweeps.append(Sweep(
            sweep_index=index, elevation_number=index + 1,
            elevation_angle_deg=elevation, nyquist_velocity_ms=32.0,
            start_status=3, end_status=2, cut_sector=0, complete=True,
            azimuth_deg=azimuth,
            elevation_deg=np.full(azimuth.size, elevation, dtype=np.float32),
            moments={
                "REF": Moment("REF", "dBZ", gates, 2125.0, 250.0, ref),
                "VEL": Moment("VEL", "m/s", gates, 2125.0, 250.0, vel),
            }))
    from pathlib import Path as _P
    return RadarVolume(
        site=RadarSite(id=site_id, name="synthetic",
                       lat_deg=float(grid.lat[j, i]),
                       lon_deg=float(grid.lon[j, i]),
                       alt_m=0.0, source="test"),
        valid_time="2026-08-01T11:30:00Z", station_id=site_id,
        volume_file=f"{site_id}20260801_113000_V06",
        volume_sha256="0" * 64, volume_bytes=8102058,
        pack_path=_P("synthetic.rdrpack"), pack_sha256="1" * 64,
        params={"moments": ["REF", "VEL"], "max_range_km": RANGE_KM},
        framing={"magic": "AR2V0006", "block_count": 1},
        sweeps=tuple(sweeps))


#: Three antennas spread across the domain so their windows differ from
#: each other as well as from the domain.
PLACED = {"AAAA": (28, 28), "BBBB": (50, 62), "CCCC": (74, 34)}


def _full_domain_window(grid, site, params):
    """What the code did before windowing: every radar covers everything."""
    return (0, grid.ny - 1, 0, grid.nx - 1)


def _contributions(grid, params):
    return [superob_volume(_volume_at(grid, site_id=s, j=j, i=i), grid,
                           params=params)
            for s, (j, i) in PLACED.items()]


# --------------------------------------------------------------------------
# The window itself.
# --------------------------------------------------------------------------


def test_window_is_a_strict_subset_of_the_domain():
    """Without this the bit-exactness test below would prove nothing."""
    grid = _grid()
    params = _params()
    volume = _volume_at(grid, site_id="AAAA", j=28, i=28)
    j0, j1, i0, i1 = horizontal_window(grid, volume.site, params)
    assert 0 <= j0 < j1 < grid.ny
    assert 0 <= i0 < i1 < grid.nx
    covered = (j1 - j0 + 1) * (i1 - i0 + 1)
    assert covered < 0.5 * grid.ny * grid.nx, (
        "the window covers most of the domain; this fixture cannot "
        "distinguish windowed from dense")


def test_window_brackets_the_range_authority():
    """The box is the reach disc plus a small margin, not a guess."""
    grid = _grid()
    params = _params()
    volume = _volume_at(grid, site_id="AAAA", j=50, i=50)
    j0, j1, i0, i1 = horizontal_window(grid, volume.site, params)
    # 40 km at 2 km cells is 20 cells each way: 41 cells, plus the margin.
    half = (j1 - j0) / 2.0
    assert 20 <= half <= 20 + superob_mod._WINDOW_MARGIN_CELLS + 1


def test_a_radar_off_the_grid_gets_a_degenerate_window():
    grid = _grid()
    params = _params()
    far = RadarSite(id="ZZZZ", name="far", lat_deg=float(grid.lat[50, 50]) + 40.0,
                    lon_deg=float(grid.lon[50, 50]), alt_m=0.0, source="test")
    assert horizontal_window(grid, far, params) == (0, 0, 0, 0)


def test_contribution_reports_its_window_for_the_receipt():
    grid = _grid()
    params = _params()
    contribution = superob_volume(
        _volume_at(grid, site_id="AAAA", j=28, i=28), grid, params=params)
    payload = contribution.window_payload()
    j0, j1, i0, i1 = contribution.window
    assert payload["j0"] == j0 and payload["j1"] == j1
    assert payload["i0"] == i0 and payload["i1"] == i1
    assert payload["nj"] == j1 - j0 + 1
    assert payload["ni"] == i1 - i0 + 1
    assert payload["cells"] == contribution.z_count.size
    assert payload["nz"] == grid.nz


# --------------------------------------------------------------------------
# The correctness claim.
# --------------------------------------------------------------------------


def test_windowed_merge_is_bitwise_identical_to_the_dense_merge(monkeypatch):
    """Windowing changes the address of a number, never the number.

    Three radars, one merge, compared array by array against the same
    merge with every window widened back to the whole domain -- which is
    exactly the code this replaced.  Bitwise, not ``allclose``: outside
    its window a radar contributed the identity of every reduction here
    (0 for the sums, -inf for a maximum, +inf for a minimum), and adding
    0.0 and taking max(x, -inf) are both exact.
    """
    grid = _grid()
    params = _params()

    windowed = merge_contributions(_contributions(grid, params), grid,
                                   params=params)
    shapes = {c.site_id: c.z_count.shape
              for c in _contributions(grid, params)}

    monkeypatch.setattr(superob_mod, "horizontal_window", _full_domain_window)
    dense = merge_contributions(_contributions(grid, params), grid,
                                params=params)

    # Guard against a vacuous pass: the windowed run must really have been
    # windowed.
    assert all(s[1] < grid.ny and s[2] < grid.nx for s in shapes.values()), (
        f"contributions were not windowed: {shapes}")

    # Reflectivity merges ACROSS radars and is not windowed at either
    # layer, so it is compared directly.
    for name in ("z_obs", "z_mask", "z_err", "z_max", "z_mean", "z_count"):
        np.testing.assert_array_equal(
            getattr(windowed, name), getattr(dense, name),
            err_msg=f"windowing changed {name}")

    # Velocity keeps one plane per radar, and those planes now cover each
    # radar's own window, so the two layouts are only commensurable
    # through the whole-domain view.  Bitwise there.
    for index in range(len(PLACED)):
        for name in ("vr_obs", "vr_mask", "vr_err", "vr_count",
                     "vr_rejected", "vr_beam_east", "vr_beam_north",
                     "vr_beam_up"):
            np.testing.assert_array_equal(
                windowed.radar_plane(name, index, ny=grid.ny, nx=grid.nx),
                dense.radar_plane(name, index, ny=grid.ny, nx=grid.nx),
                err_msg=f"windowing changed {name} for radar {index}")

    assert [r["id"] for r in windowed.radars] == [r["id"] for r in dense.radars]
    # Non-vacuity for the velocity half: the windowed planes really are
    # smaller than the dense ones.
    assert windowed.vr_obs.shape[2] < dense.vr_obs.shape[2]


def test_the_fixture_actually_produces_observations():
    """The bitwise test would also pass if both merges produced nothing."""
    grid = _grid()
    params = _params()
    merged = merge_contributions(_contributions(grid, params), grid,
                                 params=params)
    assert merged.z_mask.sum() > 0
    assert merged.vr_mask.sum() > 0
    assert merged.vr_obs.shape[0] == len(PLACED)


def test_windowing_loses_no_gates(monkeypatch):
    """Counts are conserved: the window is a superset of the support."""
    grid = _grid()
    params = _params()
    windowed = _contributions(grid, params)

    monkeypatch.setattr(superob_mod, "horizontal_window", _full_domain_window)
    dense = _contributions(grid, params)

    for w, d in zip(windowed, dense):
        assert w.site_id == d.site_id
        assert int(w.z_count.sum()) == int(d.z_count.sum())
        assert int(w.vr_count.sum()) == int(d.vr_count.sum())
        assert int(w.vr_rejected.sum()) == int(d.vr_rejected.sum())
        assert w.counts.to_payload() == d.counts.to_payload()


def test_windowing_actually_saves_the_memory_it_claims(monkeypatch):
    """The point of the exercise, as a number."""
    grid = _grid()
    params = _params()

    def _bytes(contributions):
        return sum(v.nbytes for c in contributions
                   for v in vars(c).values() if isinstance(v, np.ndarray))

    windowed = _bytes(_contributions(grid, params))
    monkeypatch.setattr(superob_mod, "horizontal_window", _full_domain_window)
    dense = _bytes(_contributions(grid, params))

    # A 40 km disc on a 200 km domain is ~17% of the cells; allow slack for
    # the margin and the bounding box being square rather than round.
    assert windowed < 0.35 * dense, (
        f"windowed {windowed} bytes vs dense {dense}: expected a large cut")


# --------------------------------------------------------------------------
# The guard, and the refusals.
# --------------------------------------------------------------------------


def test_a_window_that_under_covers_its_radar_raises(monkeypatch):
    """The one failure an analysis could not detect downstream.

    A window smaller than its radar's gates would silently drop
    observations -- the merge would succeed, the receipt would look
    healthy, and the analysis would quietly be missing data.  It raises.
    """
    grid = _grid()
    params = _params()

    def _too_small(g, site, p):
        j0, j1, i0, i1 = horizontal_window(g, site, p)
        # Halve the box about its own centre: still covers the antenna,
        # cannot cover the outer gates.
        jm, im = (j0 + j1) // 2, (i0 + i1) // 2
        return (jm - 2, jm + 2, im - 2, im + 2)

    monkeypatch.setattr(superob_mod, "horizontal_window", _too_small)
    with pytest.raises(SuperobError, match="outside the computed reach"):
        superob_volume(_volume_at(grid, site_id="AAAA", j=50, i=50), grid,
                       params=params)


def test_merge_refuses_a_window_that_does_not_fit_the_grid():
    """Previously an untested shape check; now a window check, still tested."""
    grid = _grid()
    params = _params()
    contribution = superob_volume(
        _volume_at(grid, site_id="AAAA", j=28, i=28), grid, params=params)
    # Shove the window off the eastern edge.
    contribution.i0 = grid.nx - 1
    with pytest.raises(ValueError, match="does not fit a grid"):
        merge_contributions([contribution], grid, params=params)


def test_merge_still_accepts_a_hand_built_full_domain_contribution():
    """Backward compatibility: a window at the origin spanning the grid IS
    the dense case, and downstream lanes that build one keep working."""
    grid = _grid()
    shape = (grid.nz, grid.ny, grid.nx)
    zeros = lambda: np.zeros(shape)                       # noqa: E731
    contribution = RadarContribution(
        site_id="AAAA", lat_deg=REF_LAT, lon_deg=REF_LON, alt_m=0.0,
        valid_time="2026-08-01T11:30:00Z",
        z_linear_sum=zeros(), z_count=np.zeros(shape, dtype=np.int64),
        z_max_dbz=np.full(shape, -np.inf), z_sumsq_dbz=zeros(),
        z_sum_dbz=zeros(), vr_sum=zeros(), vr_sumsq=zeros(),
        vr_count=np.zeros(shape, dtype=np.int64),
        vr_min=np.full(shape, np.inf), vr_max=np.full(shape, -np.inf),
        beam_east=zeros(), beam_north=zeros(), beam_up=zeros(),
        nyquist_min=np.full(shape, np.inf),
        vr_rejected=np.zeros(shape, dtype=np.int64),
        # The clear-air accumulator, which postdates this lane: a
        # contribution carries one whether or not any zero was established,
        # and the merge sums it like every other count.
        z0_count=np.zeros(shape, dtype=np.int64))
    assert contribution.window == (0, grid.ny - 1, 0, grid.nx - 1)
    merged = merge_contributions([contribution], grid, params=_params())
    assert merged.z_mask.sum() == 0
