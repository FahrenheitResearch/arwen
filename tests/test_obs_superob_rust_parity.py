"""The Rust superob (``rw-superob``) against the numpy reference.

The Rust library is the default route for :mod:`gpuwm.obs.superob` (and
for the region-global dealias of a sweep); the numpy body is the parity
reference, run with ``GPUWM_SUPEROB_PYTHON=1``.  Every test here runs one
scenario both ways and holds every output to the other: the accumulators,
the counters, the census, the fold-suspicion records, the CC QC and
dealias accounts, and the merged observation fields.

What "agree" means is stated per field rather than averaged away.  Integer
arrays and every counter must be identical.  Float arrays must be
identical too, except that a value computed through an elementary function
(sin, cos, atan2, pow, log10) may differ in its last bits, because numpy
dispatches those to its own SIMD kernels and Rust to the platform libm.
Measured on box K (numpy 2.5.3, AVX-512): gate placement, every count and
every sum of measured values is bit-identical; the beam-vector sums differ
by at most 1.5e-13 and the linear-Z sum by 1e-16 relative.  Such
differences are allowed at :data:`REL_TOL` relative (floor 1) and nothing
else is.

The real-volume check runs when ``GPUWM_SUPEROB_PARITY_PACKS`` names a
directory of sweep packs and ``GPUWM_SUPEROB_PARITY_GRID`` the wrfout the
observations go on.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gpuwm.obs import superob_bridge
from gpuwm.obs.cc_qc import CcQcParams
from gpuwm.obs.dealias import (ENGINE_REGION_GLOBAL, DealiasParams,
                               dealias_sweep)
from gpuwm.obs.superob import (SuperobParams, merge_contributions,
                               superob_volume)
from gpuwm.obs.sweeps import (SWEEPS_SCHEMA_CENSOR, Censor, Moment,
                              RadarSite, RadarVolume, Sweep)
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.lambert import LambertGrid

pytestmark = pytest.mark.skipif(
    superob_bridge.unavailable_reason() is not None,
    reason="rw-superob is not built on this install")

#: Largest relative difference (floor 1) allowed on a float field, for the
#: last-bit differences of the elementary functions.
REL_TOL = 1e-12

ACCUMULATORS = ("z_linear_sum", "z_count", "z0_count", "z_max_dbz",
                "z_sumsq_dbz", "z_sum_dbz", "vr_sum", "vr_sumsq", "vr_count",
                "vr_min", "vr_max", "beam_east", "beam_north", "beam_up",
                "nyquist_min", "vr_rejected")

FIELDS = ("z_obs", "z_mask", "z_err", "z_max", "z_mean", "z_count",
          "z0_mask", "z0_count", "z0_err", "vr_obs", "vr_mask", "vr_err",
          "vr_count", "vr_rejected", "vr_beam_east", "vr_beam_north",
          "vr_beam_up", "vr_beam_coherence")


def both(monkeypatch, run):
    """``(reference, rust)``: the same call down each route."""
    monkeypatch.setenv(superob_bridge.SUPEROB_PYTHON_ENV, "1")
    reference = run()
    monkeypatch.delenv(superob_bridge.SUPEROB_PYTHON_ENV)
    return reference, run()


def ulp_distance(left: np.ndarray, right: np.ndarray) -> int:
    """Largest last-place distance between two float64 arrays (NaN = NaN)."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    same = (left == right) | (np.isnan(left) & np.isnan(right))
    if same.all():
        return 0
    a = left[~same].view(np.int64)
    b = right[~same].view(np.int64)
    a = np.where(a < 0, np.int64(-(2 ** 63)) - a, a)
    b = np.where(b < 0, np.int64(-(2 ** 63)) - b, b)
    return int(np.max(np.abs(a - b)))


def assert_field(name, left, right, report=None):
    left = np.asarray(left)
    right = np.asarray(right)
    assert left.shape == right.shape, name
    assert left.dtype == right.dtype, name
    if left.dtype.kind == "f":
        same = (left == right) | (np.isnan(left) & np.isnan(right))
        if report is not None:
            report[name] = (int((~same).sum()), ulp_distance(left, right))
        scale = np.maximum(1.0, np.maximum(np.abs(np.nan_to_num(left)),
                                           np.abs(np.nan_to_num(right))))
        gap = np.where(same, 0.0, np.abs(left - right) / scale)
        assert not np.isnan(gap).any(), name
        assert gap.max(initial=0.0) <= REL_TOL, (name, gap.max())
    else:
        assert np.array_equal(left, right), name


def strip_library(account):
    """A dealias account without the library path (the routes differ)."""
    if isinstance(account, dict):
        return {k: strip_library(v) for k, v in account.items()
                if k != "library"}
    if isinstance(account, list):
        return [strip_library(v) for v in account]
    return account


def assert_contribution(reference, rust, report=None):
    for name in ACCUMULATORS:
        assert_field(name, getattr(reference, name), getattr(rust, name),
                     report)
    assert reference.window == rust.window
    assert reference.counts.to_payload() == rust.counts.to_payload()
    assert reference.fold_suspicion == rust.fold_suspicion
    assert reference.cc_qc == rust.cc_qc
    assert strip_library(reference.dealias) == strip_library(rust.dealias)
    assert reference.clear_air_source == rust.clear_air_source


def assert_observations(reference, rust, report=None):
    for name in FIELDS:
        assert_field(name, getattr(reference, name), getattr(rust, name),
                     report)
    assert reference.radar_windows == rust.radar_windows
    assert reference.counts == rust.counts
    assert reference.z_reduce == rust.z_reduce


# ---------------------------------------------------------------------------
# a synthetic volume that exercises every branch
# ---------------------------------------------------------------------------

def _grid(nx=91, ny=81, dx=3000.0, nz=24) -> TargetGrid:
    projection = LambertGrid(
        ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.2778, dx=dx, dy=dx, e_we=nx + 1, e_sn=ny + 1)
    lat, lon = projection.latlon_mass()
    terrain = 300.0 + 250.0 * np.sin(np.radians(lat * 40.0)) ** 2 \
        + 0.0 * lon
    levels = np.linspace(0.0, 1.0, nz + 1) ** 1.4 * 16000.0
    z_w = terrain[None] + levels[:, None, None] * (
        1.0 - terrain[None] / 20000.0)
    return TargetGrid.from_projection(projection, z_w=z_w, terrain_m=terrain,
                                      name="parity")


def _volume(grid, *, j, i, site_id="KPAR", seed=0, mixed_nyquist=True,
            censor=True) -> RadarVolume:
    rng = np.random.default_rng(seed)
    radials, gates = 360, 520
    azimuth = (np.arange(radials, dtype=np.float32) + np.float32(0.37)
               + rng.normal(0.0, 0.05, radials).astype(np.float32)) % 360
    az = np.radians(azimuth.astype(np.float64))[:, None]
    rng_km = (2.125 + 0.25 * np.arange(gates))[None, :]
    # A storm east of the radar, clear air elsewhere.
    storm = 58.0 * np.exp(-(((rng_km - 70.0) / 25.0) ** 2
                            + ((az - 1.4) / 0.3) ** 2))
    reflectivity = (storm - 12.0 + rng.normal(0.0, 3.0, (radials, gates)))
    codes = np.full((radials, gates), Censor.MEASURED, dtype=np.uint8)
    below = reflectivity < -5.0
    codes[below] = Censor.BELOW_THRESHOLD
    codes[rng.random((radials, gates)) < 0.01] = Censor.RANGE_FOLDED
    codes[:, 480:] = Censor.NOT_COLLECTED
    reflectivity = np.where(codes == Censor.MEASURED, reflectivity, np.nan)
    rho = np.clip(0.97 + rng.normal(0.0, 0.03, (radials, gates)), 0.2, 1.0)
    rho[:, 100:140] = 0.7                      # biota / debris band
    wind = 30.0 * np.cos(az - 0.6) + 18.0 * np.exp(
        -(((rng_km - 70.0) / 4.0) ** 2)) * np.sign(np.sin(3.0 * az))
    nyquist_row = np.where(np.arange(radials) < radials // 3, 25.51, 32.0) \
        if mixed_nyquist else np.full(radials, 26.5)
    interval = 2.0 * nyquist_row[:, None]
    folded = (wind + nyquist_row[:, None]) % interval - nyquist_row[:, None]
    folded = folded + rng.normal(0.0, 0.6, (radials, gates))
    vel_codes = np.where(np.isfinite(reflectivity), Censor.MEASURED,
                         Censor.BELOW_THRESHOLD).astype(np.uint8)
    velocity = np.where(vel_codes == Censor.MEASURED, folded, np.nan)

    def moment(product, data, codes_, unit):
        data = np.asarray(data, dtype=np.float32)
        return Moment(product, unit, gates, 2125.0, 250.0, data,
                      censor=(codes_ if censor else None))

    rho_codes = np.where(np.isfinite(rho), Censor.MEASURED,
                         Censor.NOT_COLLECTED).astype(np.uint8)

    def sweep(index, elevation, moments, nyquist):
        by_radial = None
        if nyquist == "mixed":
            by_radial = nyquist_row.astype(np.float64)
            nyquist = float(nyquist_row.min())
        return Sweep(
            sweep_index=index, elevation_number=index + 1,
            elevation_angle_deg=elevation, nyquist_velocity_ms=nyquist,
            start_status=3, end_status=2, cut_sector=0, complete=True,
            azimuth_deg=azimuth,
            elevation_deg=np.full(radials, elevation, dtype=np.float32)
            + rng.normal(0.0, 0.02, radials).astype(np.float32),
            moments=moments,
            nyquist_radials_disagree=by_radial is not None,
            nyquist_velocity_ms_by_radial=by_radial)

    sweeps = (
        # split cut: surveillance carries REF + RHO, Doppler carries VEL
        sweep(0, 0.48, {"REF": moment("REF", reflectivity, codes, "dBZ"),
                        "RHO": moment("RHO", rho, rho_codes, "")}, 9.0),
        sweep(1, 0.52, {"REF": moment("REF", reflectivity, codes, "dBZ"),
                        "VEL": moment("VEL", velocity, vel_codes, "m/s")},
              "mixed" if mixed_nyquist else 26.5),
        sweep(2, 1.45, {"REF": moment("REF", reflectivity - 3.0, codes,
                                      "dBZ"),
                        "VEL": moment("VEL", velocity, vel_codes, "m/s"),
                        "RHO": moment("RHO", rho, rho_codes, "")}, 26.5),
        sweep(3, 6.2, {"REF": moment("REF", reflectivity - 9.0, codes,
                                     "dBZ"),
                       "VEL": moment("VEL", velocity * 0.9, vel_codes,
                                     "m/s")}, None),
        sweep(4, 24.0, {"REF": moment("REF", reflectivity, codes, "dBZ")},
              26.5),
    )
    return RadarVolume(
        site=RadarSite(id=site_id, name="synthetic",
                       lat_deg=float(grid.lat[j, i]),
                       lon_deg=float(grid.lon[j, i]), alt_m=360.0,
                       source="test"),
        valid_time="2026-10-01T19:00:00Z", station_id=site_id,
        volume_file=f"{site_id}20261001_190000_V06",
        volume_sha256="0" * 64, volume_bytes=1, pack_path=Path("p.pack"),
        pack_sha256="1" * 64, params={}, framing={},
        sweeps=sweeps,
        pack_schema=SWEEPS_SCHEMA_CENSOR if censor else
        "gpuwm-obs.radar-sweeps.v1")


PARAMS = {
    "masking": lambda: SuperobParams(dealias=None),
    "region-global": lambda: SuperobParams(
        dealias=DealiasParams(engine=ENGINE_REGION_GLOBAL)),
    "region-global+cc": lambda: SuperobParams(
        dealias=DealiasParams(engine=ENGINE_REGION_GLOBAL),
        cc_qc=CcQcParams()),
    "cc+floor": lambda: SuperobParams(
        dealias=None, cc_qc=CcQcParams(rho_floor=0.6,
                                       tds_fringe_exempt=False)),
}


@pytest.mark.parametrize("arm", sorted(PARAMS))
@pytest.mark.parametrize("censor", [True, False])
def test_one_volume_agrees_on_every_output(monkeypatch, arm, censor):
    grid = _grid()
    volume = _volume(grid, j=40, i=45, censor=censor)
    params = PARAMS[arm]()
    reference, rust = both(monkeypatch, lambda: superob_volume(
        volume, grid, params=params, clear_air_from_censor=censor))
    assert rust.z_count.any() and rust.vr_count.any()
    assert_contribution(reference, rust)


@pytest.mark.parametrize("z_reduce", ["mean", "max"])
def test_the_merge_agrees_on_every_field(monkeypatch, z_reduce):
    grid = _grid()
    params = PARAMS["region-global+cc"]()
    volumes = [_volume(grid, j=j, i=i, site_id=f"K{n:03d}", seed=n)
               for n, (j, i) in enumerate(((40, 45), (10, 12), (70, 85)))]

    def run():
        contributions = [superob_volume(v, grid, params=params,
                                        clear_air_from_censor=True)
                         for v in volumes]
        return merge_contributions(contributions, grid, params=params,
                                   z_reduce=z_reduce)

    reference, rust = both(monkeypatch, run)
    assert_observations(reference, rust)
    assert rust.z_reduce == z_reduce
    chosen = rust.z_mean if z_reduce == "mean" else rust.z_max
    assert np.array_equal(rust.z_obs, chosen)


def test_the_default_reduction_is_the_linear_mean(monkeypatch):
    grid = _grid()
    params = PARAMS["masking"]()
    contribution = superob_volume(_volume(grid, j=40, i=45), grid,
                                  params=params)
    merged = merge_contributions([contribution], grid, params=params)
    assert merged.z_reduce == "mean"
    assert np.array_equal(merged.z_obs, merged.z_mean)
    has = merged.z_mask.astype(bool)
    assert (merged.z_max[has] >= merged.z_mean[has] - 1e-9).all()


def test_the_mixed_nyquist_sweep_is_unfolded_not_refused(monkeypatch):
    """The bundle-10 sector case, both routes, every plane and stat."""
    from test_obs_path_bundle10 import _wind_sweep  # noqa: PLC0415

    params = DealiasParams(engine=ENGINE_REGION_GLOBAL)
    radials = 60
    nyq = np.where(np.arange(radials) < radials // 2, 25.51, 32.0)
    folded, truth, _ = _wind_sweep(nyq, radials=radials, gates=16)
    azimuth = np.linspace(0.0, 354.0, radials)
    reference, rust = both(monkeypatch, lambda: dealias_sweep(
        folded, azimuth, float(nyq.min()), params, first_gate_m=2125.0,
        gate_spacing_m=250.0, nyquist_by_radial=nyq))
    for name in ("velocity", "state", "reason", "fold"):
        assert_field(name, getattr(reference, name), getattr(rust, name))
    assert strip_library(reference.stats) == strip_library(rust.stats)
    assert rust.stats["nyquist_distinct"] == [25.51, 32.0]
    kept = rust.state != 0
    assert np.max(np.abs(np.where(kept, rust.velocity - truth, 0.0))) < 1e-4


def test_cc_fringe_fixture_agrees(monkeypatch):
    from test_cc_qc import _fringe_volume, _grid as cc_grid  # noqa

    grid = cc_grid()
    for kwargs in ({"near_dbz": 32.0, "near_rho": 0.7},
                   {"near_dbz": 32.0, "near_rho": 0.7, "couplet": False},
                   {"near_dbz": 26.0, "near_rho": 0.4}):
        volume = _fringe_volume(grid, **kwargs)
        params = SuperobParams(dealias=None, cc_qc=CcQcParams())
        reference, rust = both(monkeypatch, lambda: superob_volume(
            volume, grid, params=params))
        assert_contribution(reference, rust)


@pytest.mark.skipif(
    not (os.environ.get("GPUWM_SUPEROB_PARITY_PACKS")
         and os.environ.get("GPUWM_SUPEROB_PARITY_GRID")),
    reason="no real sweep packs named for the parity run")
def test_real_packs_agree(monkeypatch):
    from gpuwm.obs.sweeps import read_sweep_pack  # noqa: PLC0415

    grid = TargetGrid.from_wrfout(os.environ["GPUWM_SUPEROB_PARITY_GRID"])
    packs = sorted(Path(os.environ["GPUWM_SUPEROB_PARITY_PACKS"]).glob(
        "*.pack"))[:int(os.environ.get("GPUWM_SUPEROB_PARITY_LIMIT", "4"))]
    params = PARAMS["region-global+cc"]()
    for pack in packs:
        volume = read_sweep_pack(pack)
        reference, rust = both(monkeypatch, lambda: superob_volume(
            volume, grid, params=params, clear_air_from_censor=True))
        assert_contribution(reference, rust)
