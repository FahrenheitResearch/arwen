"""The GOES CWP pack reader, the cross-grid join, and the superob gates.

Every number here is one that can be worked out by hand.  ``cod = 10``,
``cps = 15``, liquid phase gives ``(2/3) * 1.0 * 10 * 15 = 100`` g m-2;
``cod = 20``, ``cps = 30``, ice phase gives
``(2/3) * 0.917 * 20 * 30 = 366.8``.  Those two numbers, a clear-sky zero,
and a DQF-gated NaN are the whole vocabulary the gates are tested against.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.obs.goes_cwp import (CLASS_CLEAR, CLASS_ICE, CLASS_LIQUID,
                                CLASS_NONE, CwpErrorModel, GoesCwpError,
                                SuperobPolicy, grid_cwp, join_cloud_top,
                                phase_class, read_cloudtop_pack,
                                read_cwp_pack, rederive_cwp)
from gpuwm.obs.goes_pack import GoesPackError, read_goes_pack
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.lambert import LambertGrid

from goes_pack_fixtures import (CLOUDTOP_SCHEMA, CWP_SCHEMA, CWP_SCHEMA_V2,
                                sibling_block, write_cloudtop_pack,
                                write_cwp_pack)

LIQUID_CWP = 100.0
ICE_CWP = float(np.float32(np.float32(np.float32(2.0) / np.float32(3.0)
                                      * np.float32(0.917)) * np.float32(20.0))
                * np.float32(30.0))

ERRORS = CwpErrorModel(clear_g_m2=20.0, rel_liquid=0.3,
                       floor_liquid_g_m2=40.0, rel_ice=0.5,
                       floor_ice_g_m2=80.0)


def _grid(nx: int = 21, ny: int = 21, dx: float = 6000.0, nz: int = 10,
          top_m: float = 10000.0) -> TargetGrid:
    projection = LambertGrid(
        ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.2778, dx=dx, dy=dx, e_we=nx + 1, e_sn=ny + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, top_m, nz + 1), name="analytic")


def _pixels_at(grid, cells):
    """Pixel lat/lon taken from the grid's own mass points.

    A pixel written at a cell's own latitude and longitude lands in that
    cell and nowhere else, which is what makes the superob arithmetic
    below checkable by hand rather than by re-running the projection.
    """

    lat = np.array([[grid.lat[j, i] for (j, i) in cells]], dtype=np.float32)
    lon = np.array([[grid.lon[j, i] for (j, i) in cells]], dtype=np.float32)
    return lat, lon


def _scene(tmp_path, grid, cells, cod, cps, phase, **kwargs):
    lat, lon = _pixels_at(grid, cells)
    return write_cwp_pack(
        tmp_path / "cwp.goespack",
        cod=np.array([cod], np.float32), cps=np.array([cps], np.float32),
        phase=np.array([phase], np.float32), lat=lat, lon=lon,
        x_scan_rad=np.linspace(-0.02, 0.02, len(cells)),
        y_scan_rad=[0.08], **kwargs)


# ---------------------------------------------------------------------------
# the container
# ---------------------------------------------------------------------------


def test_pack_round_trip_returns_every_declared_plane(tmp_path):
    grid = _grid()
    path = _scene(tmp_path, grid, [(5, 5), (5, 6)], [10.0, 20.0],
                  [15.0, 30.0], [1.0, 4.0])
    pack = read_cwp_pack(path)
    assert pack.schema == CWP_SCHEMA
    assert pack.shape == (1, 2)
    assert list(pack.meta["plane_order"]) == ["cwp", "phase", "cod", "cps",
                                              "lat", "lon"]
    assert pack.plane("cod").tolist() == [[10.0, 20.0]]
    assert pack.plane("cwp")[0, 0] == pytest.approx(LIQUID_CWP)
    assert pack.pairing_key == ("G19", "C", "2026-08-04T18:01:17.0Z")
    # The DQF policy is the honoured-upstream half of the QC story and has
    # to survive into anything downstream that claims to know how these
    # observations were screened.
    policy = {row["product"]: row for row in pack.dqf_policy()}
    assert policy["COD"]["dqf_rule"] == "bitfield"
    # 88 = snow/sea-ice 8 | twilight 16 | glint 64, the mask the live
    # packs actually carry.  256/512 are deliberately outside it.
    assert policy["COD"]["condemn_mask"] == 88
    assert policy["ACTP"]["dqf_rule"] == "enumerated"
    assert policy["ACTP"]["condemn_mask"] is None


def test_pack_refuses_the_other_family(tmp_path):
    grid = _grid()
    lat, lon = _pixels_at(grid, [(5, 5)])
    path = write_cloudtop_pack(
        tmp_path / "ct.goespack",
        cloud_top_height_m=np.array([[9000.0]], np.float32),
        lat=lat, lon=lon, x_scan_rad=[0.0], y_scan_rad=[0.08])
    # Both families share the container on purpose, so a cloud-top pack
    # decodes perfectly and answers a different question.  The family
    # check is the only thing between that and heights being assimilated
    # as water paths.
    with pytest.raises(GoesPackError, match="caller demanded"):
        read_cwp_pack(path)
    assert read_cloudtop_pack(path).schema == CLOUDTOP_SCHEMA


@pytest.mark.parametrize("mutate,message", [
    (lambda raw: b"GPWMRDR1" + raw[8:], "magic"),
    (lambda raw: raw[:8] + (2).to_bytes(4, "little") + raw[12:], "version"),
    (lambda raw: raw[:-4], "header declares"),
    (lambda raw: raw[:-1] + bytes([raw[-1] ^ 0xFF]), "payload hashes to"),
])
def test_pack_refuses_a_damaged_container(tmp_path, mutate, message):
    grid = _grid()
    path = _scene(tmp_path, grid, [(5, 5)], [10.0], [15.0], [1.0])
    path.write_bytes(mutate(path.read_bytes()))
    with pytest.raises(GoesPackError, match=message):
        read_goes_pack(path)


def test_pack_refuses_a_status_that_is_not_ready(tmp_path):
    grid = _grid()
    path = _scene(tmp_path, grid, [(5, 5)], [10.0], [15.0], [1.0],
                  status="EMPTY")
    with pytest.raises(GoesPackError, match="only 'READY'"):
        read_goes_pack(path)


def test_phase_class_maps_the_documented_codes():
    codes = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 6.0, 1.5]])
    assert phase_class(codes).tolist() == [[
        CLASS_CLEAR, CLASS_LIQUID, CLASS_LIQUID, CLASS_ICE, CLASS_ICE,
        CLASS_NONE, CLASS_NONE, CLASS_NONE, CLASS_NONE]]


def test_rederivation_reproduces_the_written_plane(tmp_path):
    grid = _grid()
    path = _scene(tmp_path, grid, [(5, 5), (5, 6), (5, 7)],
                  [10.0, 20.0, 0.0], [15.0, 30.0, 0.0], [1.0, 4.0, 0.0])
    pack = read_cwp_pack(path)
    stated = np.asarray(pack.plane("cwp"))
    derived = rederive_cwp(pack)
    assert np.allclose(stated, derived, equal_nan=True)


def test_grid_refuses_a_pack_whose_cwp_does_not_reproduce(tmp_path):
    """A payload digest proves the bytes; it does not prove the numbers."""

    grid = _grid()
    lat, lon = _pixels_at(grid, [(5, 5)])
    path = write_cwp_pack(
        tmp_path / "lying.goespack",
        cod=np.array([[10.0]], np.float32), cps=np.array([[15.0]], np.float32),
        phase=np.array([[1.0]], np.float32), lat=lat, lon=lon,
        # A perfectly self-consistent pack whose cwp plane is simply not
        # what its own cod, cps and coefficients say it should be.
        cwp=np.array([[999.0]], np.float32),
        x_scan_rad=[0.0], y_scan_rad=[0.08])
    with pytest.raises(GoesCwpError, match="do not reproduce"):
        grid_cwp(read_cwp_pack(path), grid, error_model=ERRORS)


# ---------------------------------------------------------------------------
# the cross-grid join
# ---------------------------------------------------------------------------


def _pair(tmp_path, *, fine_x, coarse_x, heights, sibling=True,
          projection=None, scan_start=None):
    grid = _grid()
    n_fine = len(fine_x)
    cells = [(5, 5 + k) for k in range(n_fine)]
    lat, lon = _pixels_at(grid, cells)
    cwp_path = write_cwp_pack(
        tmp_path / "cwp.goespack",
        cod=np.full((1, n_fine), 10.0, np.float32),
        cps=np.full((1, n_fine), 15.0, np.float32),
        phase=np.full((1, n_fine), 1.0, np.float32),
        lat=lat, lon=lon, x_scan_rad=fine_x, y_scan_rad=[0.08])
    kwargs = {}
    if sibling:
        kwargs["sibling"] = sibling_block(cwp_path)
    if scan_start is not None:
        kwargs["scan_start"] = scan_start
    if projection is not None:
        kwargs["projection"] = projection
    ct_path = write_cloudtop_pack(
        tmp_path / "ct.goespack",
        cloud_top_height_m=np.array([heights], np.float32),
        lat=lat[:, :len(coarse_x)], lon=lon[:, :len(coarse_x)],
        x_scan_rad=coarse_x, y_scan_rad=[0.08], **kwargs)
    return grid, read_cwp_pack(cwp_path), read_cloudtop_pack(ct_path)


def test_join_nearest_maps_each_fine_pixel_to_its_own_coarse_cell(tmp_path):
    # Four fine scan angles, two coarse ones sitting exactly on the first
    # and third: nearest must give [A, A, B, B].
    _, cwp, ct = _pair(
        tmp_path, fine_x=[0.0, 0.001, 0.002, 0.003], coarse_x=[0.0, 0.002],
        heights=[9000.0, 5000.0])
    joined, receipt = join_cloud_top(cwp, ct, method="nearest")
    assert joined[0].tolist() == [9000.0, 9000.0, 5000.0, 5000.0]
    assert receipt["performed"] is True
    assert receipt["method"] == "nearest"
    assert receipt["pixels_served"] == 4
    assert receipt["source_grid"] == [1, 2]
    assert receipt["target_grid"] == [1, 4]
    assert "content_sha256 pinned and matched" in receipt[
        "sibling_digest_proof"]


def test_join_bilinear_refuses_to_blend_across_a_missing_top(tmp_path):
    """A cell touching a NaN corner has no defined value.

    NaN is "no observation"; averaging it away would manufacture a
    cloud-top height out of the absence of one.
    """

    _, cwp, ct = _pair(
        tmp_path, fine_x=[0.0, 0.001, 0.002], coarse_x=[0.0, 0.002],
        heights=[9000.0, np.nan])
    joined, receipt = join_cloud_top(cwp, ct, method="bilinear")
    assert np.isnan(joined[0]).all()
    assert receipt["method"] == "bilinear"
    assert receipt["pixels_served"] == 0


def test_join_serves_nothing_outside_the_coarse_grid(tmp_path):
    _, cwp, ct = _pair(
        tmp_path, fine_x=[0.0, 0.5], coarse_x=[0.0, 0.002],
        heights=[9000.0, 5000.0])
    joined, receipt = join_cloud_top(cwp, ct, method="nearest")
    assert joined[0, 0] == 9000.0
    assert np.isnan(joined[0, 1])
    assert receipt["pixels_outside_source_coverage"] == 1


def test_join_refuses_a_pack_from_another_scan(tmp_path):
    _, cwp, ct = _pair(
        tmp_path, fine_x=[0.0, 0.002], coarse_x=[0.0, 0.002],
        heights=[9000.0, 5000.0], sibling=False,
        scan_start="2026-08-04T18:06:17.0Z")
    with pytest.raises(GoesCwpError, match="not two halves of one scan"):
        join_cloud_top(cwp, ct)


def test_join_refuses_a_proven_pair_pointed_at_the_wrong_file(tmp_path):
    """The `sibling` digest is what makes pairing a proof, not a filename."""

    grid = _grid()
    lat, lon = _pixels_at(grid, [(5, 5)])
    other = write_cwp_pack(
        tmp_path / "other.goespack",
        cod=np.array([[11.0]], np.float32), cps=np.array([[15.0]], np.float32),
        phase=np.array([[1.0]], np.float32), lat=lat, lon=lon,
        x_scan_rad=[0.0], y_scan_rad=[0.08])
    _, cwp, _ = _pair(tmp_path, fine_x=[0.0], coarse_x=[0.0],
                      heights=[9000.0])
    ct_path = write_cloudtop_pack(
        tmp_path / "ct2.goespack",
        cloud_top_height_m=np.array([[9000.0]], np.float32),
        lat=lat, lon=lon, x_scan_rad=[0.0], y_scan_rad=[0.08],
        sibling=sibling_block(other))
    with pytest.raises(GoesCwpError, match="Refusing to join a proven pair"):
        join_cloud_top(cwp, read_cloudtop_pack(ct_path))


def test_join_refuses_two_different_geostationary_perspectives(tmp_path):
    from goes_pack_fixtures import PROJECTION

    moved = dict(PROJECTION, longitude_of_projection_origin_deg=-137.0)
    _, cwp, ct = _pair(tmp_path, fine_x=[0.0], coarse_x=[0.0],
                       heights=[9000.0], sibling=False, projection=moved)
    with pytest.raises(GoesCwpError, match="disagree on the geostationary"):
        join_cloud_top(cwp, ct)


# ---------------------------------------------------------------------------
# the superob gates and the error model
# ---------------------------------------------------------------------------


def test_superob_averages_the_cell_and_assigns_the_class_error(tmp_path):
    grid = _grid()
    # Two liquid pixels in one cell (100 and 200 g m-2), one ice pixel in
    # another, one clear pixel in a third.
    path = _scene(
        tmp_path, grid, [(5, 5), (5, 5), (7, 7), (9, 9)],
        [10.0, 20.0, 20.0, 0.0], [15.0, 15.0, 30.0, 0.0],
        [1.0, 1.0, 4.0, 0.0])
    out = grid_cwp(read_cwp_pack(path), grid, error_model=ERRORS)

    assert out.cwp_mask[5, 5] == 1
    assert out.cwp_obs[5, 5] == pytest.approx(150.0)   # (100 + 200) / 2
    assert out.cwp_count[5, 5] == 2
    assert out.cwp_class[5, 5] == CLASS_LIQUID
    # max(0.3 * 150, 40) = 45
    assert out.cwp_err[5, 5] == pytest.approx(45.0)

    assert out.cwp_obs[7, 7] == pytest.approx(ICE_CWP)
    assert out.cwp_class[7, 7] == CLASS_ICE
    # max(0.5 * 366.8, 80) = 183.4
    assert out.cwp_err[7, 7] == pytest.approx(0.5 * ICE_CWP)

    assert out.cwp_obs[9, 9] == 0.0
    assert out.cwp_class[9, 9] == CLASS_CLEAR
    assert out.cwp_err[9, 9] == pytest.approx(20.0)

    assert out.counts["observations"] == 3
    assert out.counts["observations_clear"] == 1
    assert out.counts["observations_liquid"] == 1
    assert out.counts["observations_ice"] == 1
    assert out.cwp_mask.sum() == 3


def test_superob_masks_a_cell_that_is_half_clear_and_half_ice(tmp_path):
    """The design note's rule, verbatim: that is not one observation."""

    grid = _grid()
    path = _scene(tmp_path, grid, [(5, 5), (5, 5)], [20.0, 0.0],
                  [30.0, 0.0], [4.0, 0.0])
    out = grid_cwp(read_cwp_pack(path), grid, error_model=ERRORS)
    assert out.cwp_mask[5, 5] == 0
    assert out.cwp_class[5, 5] == CLASS_NONE
    assert out.counts["cells_phase_mixed"] == 1
    assert out.counts["observations"] == 0


def test_superob_admits_a_mixed_cell_when_uniformity_is_relaxed(tmp_path):
    grid = _grid()
    path = _scene(tmp_path, grid, [(5, 5), (5, 5), (5, 5), (5, 5)],
                  [10.0, 10.0, 10.0, 0.0], [15.0, 15.0, 15.0, 0.0],
                  [1.0, 1.0, 1.0, 0.0])
    policy = SuperobPolicy(phase_uniform_fraction=0.75)
    out = grid_cwp(read_cwp_pack(path), grid, error_model=ERRORS,
                   policy=policy)
    assert out.cwp_mask[5, 5] == 1
    # The areal mean of every valid pixel -- three at 100, one clear zero.
    assert out.cwp_obs[5, 5] == pytest.approx(75.0)
    # ...under the DOMINANT class's error model.
    assert out.cwp_class[5, 5] == CLASS_LIQUID
    assert out.cwp_err[5, 5] == pytest.approx(40.0)   # max(0.3*75, 40)


def test_dqf_gated_pixels_are_no_observation_and_are_counted(tmp_path):
    """The bridge condemns upstream; NaN arrives and must stay a hole."""

    grid = _grid()
    path = _scene(tmp_path, grid, [(5, 5), (7, 7)], [10.0, np.nan],
                  [15.0, np.nan], [1.0, np.nan])
    out = grid_cwp(read_cwp_pack(path), grid, error_model=ERRORS)
    assert out.cwp_mask[5, 5] == 1
    assert out.cwp_mask[7, 7] == 0
    assert out.counts["pixels_no_phase"] == 1
    assert out.counts["observations"] == 1
    # ...and the rule that condemned it travels into the receipt.
    products = {row["product"] for row in out.provenance["dqf_policy"]}
    assert products == {"COD", "CPS", "ACTP"}


def test_min_valid_fraction_drops_a_mostly_condemned_cell(tmp_path):
    grid = _grid()
    path = _scene(tmp_path, grid, [(5, 5), (5, 5), (5, 5)],
                  [10.0, np.nan, np.nan], [15.0, np.nan, np.nan],
                  [1.0, np.nan, np.nan])
    out = grid_cwp(read_cwp_pack(path), grid, error_model=ERRORS,
                   policy=SuperobPolicy(min_valid_fraction=0.5))
    assert out.cwp_mask[5, 5] == 0
    assert out.counts["cells_below_min_valid_fraction"] == 1


def test_cloudy_columns_are_centred_at_the_retrieved_top(tmp_path):
    grid = _grid()
    cells = [(5, 5), (9, 9)]
    lat, lon = _pixels_at(grid, cells)
    cwp_path = write_cwp_pack(
        tmp_path / "cwp.goespack",
        cod=np.array([[20.0, 0.0]], np.float32),
        cps=np.array([[30.0, 0.0]], np.float32),
        phase=np.array([[4.0, 0.0]], np.float32),
        lat=lat, lon=lon, x_scan_rad=[0.0, 0.002], y_scan_rad=[0.08])
    ct_path = write_cloudtop_pack(
        tmp_path / "ct.goespack",
        cloud_top_height_m=np.array([[9000.0, 9000.0]], np.float32),
        lat=lat, lon=lon, x_scan_rad=[0.0, 0.002], y_scan_rad=[0.08],
        sibling=sibling_block(cwp_path))
    cwp = read_cwp_pack(cwp_path)
    joined, receipt = join_cloud_top(cwp, read_cloudtop_pack(ct_path))
    out = grid_cwp(cwp, grid, error_model=ERRORS, cloud_top_m=joined,
                   join_receipt=receipt)
    # z_w is 0..10000 in ten 1000 m layers, so 9000 m is level 9 and the
    # 3000 m fallback is level 3.
    assert out.obs_level[5, 5] == 9
    assert out.cloud_top_height_m[5, 5] == pytest.approx(9000.0)
    # A clear-sky zero has no retrieved top; it takes the fallback.
    assert out.obs_level[9, 9] == 3
    assert np.isnan(out.cloud_top_height_m[9, 9])
    assert out.counts["observations_at_retrieved_top"] == 1
    assert out.counts["observations_at_fallback_height"] == 1
    assert out.provenance["join"]["method"] == "nearest"


def test_a_joined_field_without_its_receipt_is_refused(tmp_path):
    """An unrecorded interpolation is the thing the ruling exists to stop."""

    grid = _grid()
    path = _scene(tmp_path, grid, [(5, 5)], [20.0], [30.0], [4.0])
    pack = read_cwp_pack(path)
    with pytest.raises(GoesCwpError, match="without the receipt"):
        grid_cwp(pack, grid, error_model=ERRORS,
                 cloud_top_m=np.array([[9000.0]]))


def test_no_join_is_recorded_rather_than_left_silent(tmp_path):
    grid = _grid()
    path = _scene(tmp_path, grid, [(5, 5)], [20.0], [30.0], [4.0])
    out = grid_cwp(read_cwp_pack(path), grid, error_model=ERRORS)
    assert out.provenance["join"]["performed"] is False
    assert "fallback_placement_agl_m" in out.provenance["join"]["consequence"]


def test_error_model_refuses_to_trust_ice_more_than_liquid():
    with pytest.raises(GoesCwpError, match="PROVISIONAL"):
        CwpErrorModel(clear_g_m2=20.0, rel_liquid=0.5,
                      floor_liquid_g_m2=40.0, rel_ice=0.3,
                      floor_ice_g_m2=80.0).validate()


def test_error_model_refuses_a_zero_sigma():
    with pytest.raises(GoesCwpError, match="infinite weight"):
        CwpErrorModel(clear_g_m2=0.0, rel_liquid=0.3, floor_liquid_g_m2=40.0,
                      rel_ice=0.5, floor_ice_g_m2=80.0).validate()


def test_the_error_model_names_itself_uncalibrated():
    payload = ERRORS.to_payload()
    assert payload["calibration"] == "UNCALIBRATED"
    # No inflation requested: the row is present, off, and says the bits
    # it would have used rather than going silent.
    inflation = payload["thin_thick_inflation"]
    assert inflation["applied"] is False
    assert inflation["thin_bit"] == 256 and inflation["thick_bit"] == 512
    assert inflation["calibration"] == "UNCALIBRATED"


def test_inflation_factors_below_one_are_refused():
    with pytest.raises(GoesCwpError, match="opposite of what the flag says"):
        CwpErrorModel(clear_g_m2=20.0, rel_liquid=0.3,
                      floor_liquid_g_m2=40.0, rel_ice=0.5,
                      floor_ice_g_m2=80.0, thin_inflation=0.5).validate()


def test_a_v1_pack_refuses_the_thin_thick_inflation(tmp_path):
    """The spec row a v1 pack cannot serve is a refusal, not a no-op.

    Silently not inflating looks exactly like not needing to, which is the
    failure mode the schema bump exists to remove.
    """

    grid = _grid()
    path = _scene(tmp_path, grid, [(5, 5)], [10.0], [15.0], [1.0])
    inflating = CwpErrorModel(clear_g_m2=20.0, rel_liquid=0.3,
                              floor_liquid_g_m2=40.0, rel_ice=0.5,
                              floor_ice_g_m2=80.0, thick_inflation=2.0)
    with pytest.raises(GoesCwpError, match="no per-pixel DQF plane"):
        grid_cwp(read_cwp_pack(path), grid, error_model=inflating)
    # ...and the same pack is fine when nothing is asked of it.
    assert grid_cwp(read_cwp_pack(path), grid,
                    error_model=ERRORS).cwp_mask.sum() == 1


def test_a_v2_pack_inflates_the_thin_and_thick_pixels(tmp_path):
    grid = _grid()
    # Four cells: clean, thin (256), thick (512), and both.
    cells = [(5, 5), (5, 6), (5, 7), (5, 8)]
    lat, lon = _pixels_at(grid, cells)
    dqf = np.array([[0.0, 256.0, 512.0, 768.0]], np.float32)
    path = write_cwp_pack(
        tmp_path / "v2.goespack",
        cod=np.full((1, 4), 10.0, np.float32),
        cps=np.full((1, 4), 15.0, np.float32),
        phase=np.full((1, 4), 1.0, np.float32), lat=lat, lon=lon,
        x_scan_rad=np.linspace(-0.02, 0.02, 4), y_scan_rad=[0.08],
        schema=CWP_SCHEMA_V2, dqf_planes={"COD": dqf, "CPS": dqf,
                                          "ACTP": np.zeros((1, 4), np.float32)})
    inflating = CwpErrorModel(clear_g_m2=20.0, rel_liquid=0.3,
                              floor_liquid_g_m2=40.0, rel_ice=0.5,
                              floor_ice_g_m2=80.0, thin_inflation=2.0,
                              thick_inflation=3.0)
    out = grid_cwp(read_cwp_pack(path), grid, error_model=inflating)
    # Every cell is liquid CWP 100 -> max(0.3*100, 40) = 40 before inflation.
    assert out.cwp_err[5, 5] == pytest.approx(40.0)
    assert out.cwp_err[5, 6] == pytest.approx(80.0)     # thin  x2
    assert out.cwp_err[5, 7] == pytest.approx(120.0)    # thick x3
    assert out.cwp_err[5, 8] == pytest.approx(240.0)    # both  x6
    receipt = out.provenance["error_model"]["thin_thick_inflation"]
    assert receipt["applied"] is True
    assert receipt["counts"]["pixels_thin"] == 2      # 256 and 768
    assert receipt["counts"]["pixels_thick"] == 2     # 512 and 768
    assert receipt["counts"]["pixels_both"] == 1
    assert out.counts["cells_error_inflated"] == 3
    assert out.provenance["pack_schema_version"] == 2
    assert out.provenance["per_pixel_dqf_available"] is True


def test_an_unreadable_dqf_pixel_is_never_read_as_a_bitfield(tmp_path):
    """NaN DQF is not zero.  astype(uint16) on it returns garbage."""

    grid = _grid()
    cells = [(5, 5), (5, 6)]
    lat, lon = _pixels_at(grid, cells)
    dqf = np.array([[np.nan, 512.0]], np.float32)
    path = write_cwp_pack(
        tmp_path / "v2nan.goespack",
        cod=np.full((1, 2), 10.0, np.float32),
        cps=np.full((1, 2), 15.0, np.float32),
        phase=np.full((1, 2), 1.0, np.float32), lat=lat, lon=lon,
        x_scan_rad=[-0.01, 0.01], y_scan_rad=[0.08],
        schema=CWP_SCHEMA_V2, dqf_planes={"COD": dqf, "CPS": dqf,
                                          "ACTP": np.zeros((1, 2), np.float32)})
    inflating = CwpErrorModel(clear_g_m2=20.0, rel_liquid=0.3,
                              floor_liquid_g_m2=40.0, rel_ice=0.5,
                              floor_ice_g_m2=80.0, thick_inflation=3.0)
    out = grid_cwp(read_cwp_pack(path), grid, error_model=inflating)
    assert out.cwp_err[5, 5] == pytest.approx(40.0)     # unreadable -> 1.0x
    assert out.cwp_err[5, 6] == pytest.approx(120.0)
    counts = out.provenance["error_model"]["thin_thick_inflation"]["counts"]
    # A per-pixel union across the DCOMP products, not a per-product sum:
    # one pixel is unreadable, in both products.
    assert counts["pixels_dqf_unreadable"] == 1
