"""Radar latent heating builder on the card, graded against NOAA's code.

The fixture ``tests/data/radar_tten_oracle_fixture.npz`` holds inputs and
the answers of NOAA-EMC/HRRR v4.1.21's own ``calc_pbl_height``,
``build_missing_REFcone`` and ``radar_ref2tten``, compiled unchanged with
gfortran ``-O2 -ffp-contract=off`` and called in the driver's order with its
constants (``tools/radar_tten_proof``: ``build.sh``, ``harness.f90``,
``cases.py --fixture``).  144 columns of 50 levels, every recipe of the
proof's branch case on at least three interior columns, with the
convection-only switch on and off.  The full proof (64 x 64 x 50 and three
200 x 200 x 50 random fields) is ``tools/radar_tten_proof/compare.py``.

The smoother is also checked against a NumPy transcription of
``smooth.f90`` written here, on a field whose edges are not zero (inside
``radar_ref2tten`` they always are).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

FIXTURE = Path(__file__).resolve().parent / "data" / \
    "radar_tten_oracle_fixture.npz"

pytestmark = [pytest.mark.gpu, requires_gpu]


def _inputs(cp, data):
    return [cp.asarray(data[f"in_{name}"]) for name in
            ("ref", "t", "p", "q", "h")]


@pytest.mark.parametrize("convection_only", [1, 0])
def test_the_builder_is_noaas_routines_bit_for_bit(convection_only):
    import cupy as cp
    from gpuwm.da import radar_tten

    data = np.load(FIXTURE)
    ref, theta, p, q, h = _inputs(cp, data)
    keep = {}
    slot, receipt = radar_tten.build_tendency(
        ref, theta, p, q, h,
        radar_tten.RadarTtenConfig(convection_only=bool(convection_only)),
        intermediates=keep)
    prefix = f"out{convection_only}_"
    np.testing.assert_array_equal(
        cp.asnumpy(keep["pblh"]).view(np.int32),
        data[prefix + "pblh"].view(np.int32))
    np.testing.assert_array_equal(
        cp.asnumpy(keep["ref_cone"]).view(np.int32),
        data[prefix + "refcone"].view(np.int32))
    want = data[prefix + "tten"]
    got = cp.asnumpy(slot)
    np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32))
    # the fixture reaches what it is meant to reach
    below = want[:-1]
    assert np.count_nonzero(below == -20.0) > 0
    assert np.count_nonzero(below == 0.0) > 0
    assert np.count_nonzero(below > 0.0) > 0
    assert set(np.unique(want[-1])) == {-10.0, 0.0, 1.0}
    assert receipt["points_heated"] == int(np.count_nonzero(
        (below > 0.0) & (below <= 1.0)))
    assert receipt["points_no_coverage"] == int(np.count_nonzero(
        below == -20.0))
    # the input is not modified
    np.testing.assert_array_equal(cp.asnumpy(ref), data["in_ref"])


def test_the_stratiform_switch_hands_columns_back_to_the_model():
    """radar_ref2tten.f90:315-328: a column flagged 'convection nearby'
    without probable convection becomes no coverage with flag -10."""
    data = np.load(FIXTURE)
    on, off = data["out1_tten"], data["out0_tten"]
    handed_back = (off[-1] == 1.0) & (on[-1] == -10.0)
    assert handed_back.any()
    assert np.all(on[1:-1][:, handed_back] == -20.0)


def _numpy_smooth(field, passes, s=0.5):
    """smooth.f90, transcribed: interior from the unsmoothed field, then
    the four edges in place, in index order."""
    f = np.array(field, dtype=np.float64)
    s1, s2, s3, s4, s5 = (0.25 * s * s, 0.5 * s * (1.0 - s),
                          (1.0 - s) * (1.0 - s), (1.0 - s), 0.5 * s)
    for _ in range(passes):
        g = f.copy()
        sum1 = ((f[2:, :-2] + f[:-2, :-2]) + f[2:, 2:]) + f[:-2, 2:]
        sum2 = ((f[2:, 1:-1] + f[1:-1, 2:]) + f[:-2, 1:-1]) + f[1:-1, :-2]
        g[1:-1, 1:-1] = (s1 * sum1 + s2 * sum2) + s3 * f[1:-1, 1:-1]
        ny, nx = g.shape
        for i in range(1, nx - 1):
            g[0, i] = s4 * g[0, i] + s5 * (g[0, i - 1] + g[0, i + 1])
            g[-1, i] = s4 * g[-1, i] + s5 * (g[-1, i - 1] + g[-1, i + 1])
        for j in range(1, ny - 1):
            g[j, 0] = s4 * g[j, 0] + s5 * (g[j - 1, 0] + g[j + 1, 0])
            g[j, -1] = s4 * g[j, -1] + s5 * (g[j - 1, -1] + g[j + 1, -1])
        f = g
    return f


@pytest.mark.parametrize("passes", [1, 2, 5])
def test_the_smoother_is_smooth_f90_with_its_edge_recurrence(passes):
    import cupy as cp
    from gpuwm.da import radar_tten

    rng = np.random.default_rng(11)
    field = rng.uniform(-1.0, 1.0, size=(3, 17, 23))
    got = cp.asnumpy(radar_tten.smooth(cp.asarray(field), passes=passes))
    for k in range(field.shape[0]):
        want = _numpy_smooth(field[k], passes)
        np.testing.assert_array_equal(got[k].view(np.int64),
                                      want.view(np.int64))


def _numpy_vinterp(mosaic, h, zh, levels):
    """vinterp_radar_ref.f90 :84-148, transcribed, corner quirk included."""
    nz, ny, nx = h.shape
    out = np.full(h.shape, -99999.0, np.float32)
    for k in range(nz):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                hg = float(np.float32(h[k, j, i] + zh[j, i]))
                if not (levels[0] <= hg < levels[-1]):
                    continue
                ilvl = 0
                for m in range(len(levels) - 1):
                    if levels[m] <= hg < levels[m + 1]:
                        ilvl = m
                up = float(mosaic[ilvl + 1, j, i])
                down = float(mosaic[ilvl, j, i])
                if abs(up) < 90.0 and abs(down) < 90.0:
                    w = (hg - levels[ilvl]) / (levels[ilvl + 1] - levels[ilvl])
                    value = (1.0 - w) * down + w * up
                elif abs(up + 99.0) < 0.1 or abs(down + 99.0) < 0.1:
                    value = -99.0
                else:
                    value = -99999.0
                out[k, j, i] = np.float32(max(-99999.0, value))
        out[k, 0, 1:-1] = out[k, 1, 1:-1]
        out[k, -1, 1:-1] = out[k, -2, 1:-1]
        out[k, 1:-1, 0] = out[k, 1:-1, 1]
        out[k, 1:-1, -1] = out[k, 1:-1, -2]
        out[k, -1, -1] = out[k, -2, -2]
        out[k, 0, -1] = out[k, 1, -2]
        out[k, -1, 0] = out[k, 1, 1]
    return out


@pytest.mark.parametrize("levels", [21, 31, 33])
def test_the_mosaic_interpolation_is_vinterp_radar_ref(levels):
    import cupy as cp
    from gpuwm.da import radar_tten

    rng = np.random.default_rng(levels)
    nz, ny, nx = 14, 8, 9
    zh = rng.uniform(0.0, 1200.0, size=(ny, nx)).astype(np.float32)
    h = ((np.arange(nz)[:, None, None] + 0.5) * 1500.0
         + rng.uniform(-60.0, 60.0, size=(nz, ny, nx))).astype(np.float32)
    zh[3, 4] = 0.0
    table = np.asarray(radar_tten.MOSAIC_LEVELS_KM[levels], np.float32)
    h[:5, 3, 4] = table[:5] * 1000.0          # exactly on mosaic levels
    mosaic = rng.uniform(-30.0, 80.0, size=(levels, ny, nx))
    pick = rng.integers(0, 6, size=mosaic.shape)
    mosaic = np.where(pick == 0, -99.0, mosaic)
    mosaic = np.where(pick == 1, -999.0, mosaic)
    mosaic = np.where(pick == 2, 95.0, mosaic).astype(np.float32)
    got = cp.asnumpy(radar_tten.vinterp_mosaic(
        cp.asarray(mosaic), cp.asarray(h), cp.asarray(zh)))
    want = _numpy_vinterp(mosaic, h, zh,
                          [float(v) * 1000.0 for v in table.astype(np.float64)])
    np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32))
    assert np.count_nonzero(np.abs(want) < 90.0) > 0
    assert np.count_nonzero(want == -99.0) > 0
    assert want[0, 0, 0] == -99999.0          # NOAA's untouched corner


def test_a_mosaic_with_an_unknown_level_count_is_refused():
    import cupy as cp
    from gpuwm.da import radar_tten

    h = cp.zeros((6, 5, 5), dtype=cp.float32)
    with pytest.raises(radar_tten.RadarTtenError, match="level table"):
        radar_tten.vinterp_mosaic(cp.zeros((30, 5, 5), dtype=cp.float32), h,
                                  cp.zeros((5, 5), dtype=cp.float32))


def _document(z_obs, z_mask, z0_mask=None, source="finite_below_floor"):
    variables = {"z_obs": z_obs, "z_mask": z_mask.astype(np.int8)}
    document = {"variables": variables}
    if z0_mask is not None:
        variables["z0_mask"] = z0_mask.astype(np.int8)
        document["clear_air_source"] = source
    return document


def test_the_document_adapter_writes_noaas_three_values():
    import cupy as cp
    from gpuwm.da import radar_tten

    shape = (5, 4, 6)
    z = np.full(shape, 12.5, np.float32)
    echo = np.zeros(shape, bool)
    echo[2, 1, 2] = True
    clear = np.zeros(shape, bool)
    clear[3] = True
    ref, provenance = radar_tten.reflectivity_from_document(
        _document(z, echo, clear))
    ref = cp.asnumpy(ref)
    assert ref[2, 1, 2] == np.float32(12.5)
    assert np.all(ref[3] == -99.0)
    rest = ~(echo | clear)
    assert np.all(ref[rest] == -99999.0)
    assert provenance["echo_points"] == 1
    assert provenance["clear_air_points"] == clear.sum()

    ref, provenance = radar_tten.reflectivity_from_document(
        _document(z, echo))
    ref = cp.asnumpy(ref)
    assert np.all(ref[~echo] == -99999.0)
    assert provenance["clear_air"].startswith("absent")


def test_the_document_adapter_refuses_what_it_cannot_read_safely():
    from gpuwm.da import radar_tten

    shape = (5, 4, 6)
    z = np.full(shape, 30.0, np.float32)
    echo = np.zeros(shape, bool)
    echo[1, 1, 1] = True
    with pytest.raises(radar_tten.RadarTtenError, match="range|regime|reads"):
        radar_tten.reflectivity_from_document(
            _document(z, echo, ~echo, source="range_folded"))
    with pytest.raises(radar_tten.RadarTtenError, match="both"):
        radar_tten.reflectivity_from_document(_document(z, echo, echo))
    bad = z.copy()
    bad[1, 1, 1] = np.nan
    with pytest.raises(radar_tten.RadarTtenError, match="finite"):
        radar_tten.reflectivity_from_document(_document(bad, echo))


def test_host_arrays_are_refused_not_converted():
    from gpuwm.da import radar_tten

    data = np.load(FIXTURE)
    with pytest.raises(radar_tten.RadarTtenError, match="device array"):
        radar_tten.build_tendency(*(data[f"in_{n}"] for n in
                                    ("ref", "t", "p", "q", "h")))


def test_a_grid_too_small_for_noaas_loops_is_refused():
    import cupy as cp
    from gpuwm.da import radar_tten

    small = cp.zeros((3, 5, 5), dtype=cp.float32)
    with pytest.raises(radar_tten.RadarTtenError, match="smaller"):
        radar_tten.build_tendency(small, small, small, small, small)


def test_the_background_is_read_off_the_model_state():
    import cupy as cp
    from gpuwm.core import constants as c
    from gpuwm.da import radar_tten
    from test_radar_tten_forcing import _moist_state

    state, cfg = _moist_state()
    bg = radar_tten.background_from_state(state)
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    np.testing.assert_array_equal(cp.asnumpy(bg["theta"]),
                                  cp.asnumpy(thb + state.thp))
    np.testing.assert_allclose(cp.asnumpy(bg["pressure_hpa"]),
                               cp.asnumpy(state.p) / 100.0, rtol=1e-7)
    h = cp.asnumpy(bg["height_agl_m"]).astype(np.float64)
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    z = cp.asnumpy((phb + state.php) / np.float32(c.G)).astype(np.float64)
    want = 0.5 * (z[:-1] + z[1:]) - z[0]
    np.testing.assert_allclose(h, np.broadcast_to(want, h.shape),
                               rtol=0, atol=0.05)
    assert h.min() > 0.0
