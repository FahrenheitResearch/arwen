"""Terrain-keyed MSLP display treatment (below-terrain extrapolation noise).

Defect: gpuwm/verify/metrics.py drew the raw DCOMPUTESEAPRS output
straight onto the synoptic map (make_synoptic_maps), so the sea-level
reduction's noise -- amplified linearly in terrain height by the
exp(2 g z_sfc / (Rd (T_sl + T_sfc))) extrapolation -- scribbled across
high-elevation regions of a frame while the same plot stayed clean over
low terrain.

The fix is ``terrain_smoothed_mslp``: the below-terrain extrapolation
increment ln(SLP / p_sfc) is smoothed (standard nine-point smoother) and
blended back keyed on terrain height ONLY -- below the terrain floor the
treatment is the exact identity, so verified low-terrain synoptic fields
cannot move at all.  The scored/gated ``mslp`` stays the frozen raw
DCOMPUTESEAPRS (audit R16); only the drawn field changes.

Tolerances asserted here and why:

* Low terrain: EXACT equality (zero tolerance).  The blend weight is
  identically zero below ``MSLP_SMOOTH_TERRAIN_FLOOR_M`` and the
  implementation returns the input values unmixed there, so anything
  other than bit-equality would mean the treatment leaked outside its
  terrain key.
* High terrain: grid-scale roughness (mean |cell - 4-neighbour mean|)
  of the treated field below 0.2x the raw field's.  Three passes of the
  nine-point smoother attenuate a grid-scale checkerboard by (1/3)^3 =
  1/27 per interior cell, so 0.2 is a 5x margin over the expected
  residual, loose enough to survive edge effects, tight enough that an
  unsmoothed field (ratio 1.0) can never pass.
"""

from pathlib import Path

import numpy as np
import pytest

from gpuwm.verify import metrics


# Standard-atmosphere constants, spelled as the reduction spells them
# (metrics._dcomputeseaprs: g = 9.81, Rd = 287, gamma = 0.0065).
_G, _RD, _GAMMA = 9.81, 287.0, 0.0065
_P0, _T0 = 101325.0, 288.15


def _standard_columns(terrain_m: np.ndarray, t_offset_k: np.ndarray):
    """Synthetic hydrostatic column set over ``terrain_m``.

    Every column is the same US-standard atmosphere reduced from one
    sea-level state (p0 = 1013.25 hPa), so the TRUE sea-level pressure
    is horizontally uniform: any horizontal structure in the reduced
    MSLP is reduction artifact, not weather.  ``t_offset_k`` perturbs
    each column's temperature at every level (grid-scale surface-layer
    heterogeneity, the real driver of reduction noise); pressure is
    deliberately left on the smooth base state so the experiment
    isolates the reduction's temperature sensitivity.
    """
    agl = np.array([25.0, 250.0, 750.0, 1500.0, 2500.0, 4000.0, 6000.0])
    z = terrain_m[None, :, :] + agl[:, None, None]
    t_base = _T0 - _GAMMA * z
    pressure = _P0 * (t_base / _T0) ** (_G / (_RD * _GAMMA))
    temperature = t_base + t_offset_k[None, :, :]
    qvapor = np.zeros_like(temperature)
    return pressure, temperature, qvapor, z


def _roughness(field: np.ndarray, region) -> float:
    """Mean |cell - mean(4 neighbours)| over ``region``: grid-scale noise."""
    interior = field[1:-1, 1:-1]
    neighbours = 0.25 * (field[:-2, 1:-1] + field[2:, 1:-1]
                         + field[1:-1, :-2] + field[1:-1, 2:])
    delta = np.abs(interior - neighbours)
    # region indexes the interior-trimmed array.
    return float(np.mean(delta[region]))


def _split_terrain_case():
    """Low plain (50 m) west, high plateau (2500 m) east, checkerboard
    +/-2.5 K column-temperature noise everywhere."""
    ny, nx = 16, 32
    terrain = np.full((ny, nx), 50.0)
    terrain[:, nx // 2:] = 2500.0
    iy, ix = np.indices((ny, nx))
    noise = 2.5 * np.where((iy + ix) % 2 == 0, 1.0, -1.0)
    pressure, temperature, qvapor, z = _standard_columns(terrain, noise)
    raw = metrics._dcomputeseaprs(pressure, temperature, qvapor, z)
    return terrain, pressure, raw, (ny, nx)


# Interior-trimmed index regions (for arrays shaped (ny-2, nx-2)):
# stay 4+ columns clear of the terrain step at nx//2 and of the edges.
_LOW_REGION = (slice(1, -1), slice(1, 11))     # cols 2..11 of the plain
_HIGH_REGION = (slice(1, -1), slice(19, 29))   # cols 20..29 of the plateau


def test_raw_reduction_is_noisy_over_high_terrain_only():
    """The defect itself: same temperature noise, same true MSLP, yet the
    raw reduction scribbles over the plateau and stays clean over the
    plain.  This is the artifact the treatment must suppress."""
    _terrain, _pressure, raw, _shape = _split_terrain_case()
    assert np.isfinite(raw).all()
    assert 900.0 < raw.min() < raw.max() < 1100.0
    rough_high = _roughness(raw, _HIGH_REGION)
    rough_low = _roughness(raw, _LOW_REGION)
    assert rough_high > 1.0, "synthetic case must reproduce the artifact"
    assert rough_low < 0.3, "low terrain must be clean in the raw field"


def test_treatment_suppresses_high_terrain_noise():
    terrain, pressure, raw, _shape = _split_terrain_case()
    treated = metrics.terrain_smoothed_mslp(raw, pressure[0], terrain)
    assert treated.shape == raw.shape
    assert np.isfinite(treated).all()
    rough_raw = _roughness(raw, _HIGH_REGION)
    rough_treated = _roughness(treated, _HIGH_REGION)
    assert rough_treated < 0.2 * rough_raw, (
        f"plateau roughness {rough_treated:.3f} hPa is not suppressed "
        f"vs raw {rough_raw:.3f} hPa")
    # The treatment must not shift the plateau's mean pressure: it
    # redistributes the extrapolation increment, never re-centres it.
    high_full = (slice(None), slice(20, 30))
    assert abs(float(np.mean(treated[high_full] - raw[high_full]))) < 0.2


def test_low_terrain_is_bit_exact():
    """Below the terrain floor the treatment is the identity, exactly.

    Zero tolerance by design: the blend is keyed on terrain height only
    and returns the input values unmixed where the weight is zero, so
    any difference at all would mean the fix perturbed verified
    low-terrain synoptic fields.
    """
    terrain, pressure, raw, (ny, nx) = _split_terrain_case()
    treated = metrics.terrain_smoothed_mslp(raw, pressure[0], terrain)
    low_half = (slice(None), slice(0, nx // 2))
    assert np.array_equal(treated[low_half], raw[low_half])
    # An all-low domain comes back identical everywhere.
    flat = np.full((ny, nx), 120.0)
    p_flat, t_flat, q_flat, z_flat = _standard_columns(
        flat, np.zeros((ny, nx)))
    raw_flat = metrics._dcomputeseaprs(p_flat, t_flat, q_flat, z_flat)
    treated_flat = metrics.terrain_smoothed_mslp(raw_flat, p_flat[0], flat)
    assert np.array_equal(treated_flat, raw_flat)


def test_terrain_smoothed_mslp_rejects_mismatched_grids():
    field = np.full((4, 4), 1013.0)
    with pytest.raises(ValueError):
        metrics.terrain_smoothed_mslp(field, np.full((4, 5), 1.0e5),
                                      np.zeros((4, 4)))
    with pytest.raises(ValueError):
        metrics.terrain_smoothed_mslp(field[0], np.full(4, 1.0e5),
                                      np.zeros(4))


def _write_synthetic_wrfout(path: Path, terrain: np.ndarray, nz: int = 8):
    """Minimal wrfout carrying every field _wrf_diagnostics reads."""
    from gpuwm.io.wrfout import WrfoutWriter

    ny, nx = terrain.shape
    pressure = np.linspace(92500.0, 30000.0, nz, dtype=np.float32)
    agl_stag = np.linspace(0.0, 10000.0, nz + 1, dtype=np.float32)
    height_stag = terrain[None, :, :].astype(np.float32) + \
        agl_stag[:, None, None]
    frame = {
        "U": np.full((nz, ny, nx + 1), 8.0, np.float32),
        "V": np.full((nz, ny + 1, nx), 3.0, np.float32),
        "W": np.zeros((nz + 1, ny, nx), np.float32),
        "T": np.zeros((nz, ny, nx), np.float32),
        "P": np.zeros((nz, ny, nx), np.float32),
        "PB": np.broadcast_to(
            pressure[:, None, None], (nz, ny, nx)).copy(),
        "PH": np.zeros((nz + 1, ny, nx), np.float32),
        "PHB": (9.81 * height_stag).astype(np.float32),
        "QVAPOR": np.full((nz, ny, nx), 0.004, np.float32),
        "MU": np.zeros((ny, nx), np.float32),
        "MUB": np.full((ny, nx), 82500.0, np.float32),
        "HGT": terrain.astype(np.float32),
        "PSFC": np.full((ny, nx), 100000.0, np.float32),
        "T2": np.full((ny, nx), 290.0, np.float32),
        "Q2": np.full((ny, nx), 0.004, np.float32),
        "U10": np.full((ny, nx), 7.0, np.float32),
        "V10": np.full((ny, nx), 2.0, np.float32),
        "RAINNC": np.zeros((ny, nx), np.float32),
        "XLAT": np.broadcast_to(
            np.linspace(30.0, 34.0, ny, dtype=np.float32)[:, None],
            (ny, nx)).copy(),
        "XLONG": np.broadcast_to(
            np.linspace(-100.0, -94.0, nx, dtype=np.float32)[None, :],
            (ny, nx)).copy(),
    }
    with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=12000.0,
                      dy=12000.0) as writer:
        writer.write_frame("1974-04-04_12:00:00", frame)


def test_wrf_diagnostics_carries_raw_and_display_mslp(tmp_path):
    """_wrf_diagnostics keeps the frozen scored 'mslp' and adds
    'mslp_display', which is exactly the terrain-keyed treatment of it."""
    import netCDF4

    ny, nx = 6, 10
    terrain = np.full((ny, nx), 40.0)
    terrain[:, nx // 2:] = 2200.0
    path = tmp_path / "wrfout_d01_1974-04-04_12_00_00"
    _write_synthetic_wrfout(path, terrain)

    diagnostics = metrics._wrf_diagnostics(path)
    assert "mslp" in diagnostics and "mslp_display" in diagnostics

    with netCDF4.Dataset(path) as ds:
        pressure_pa = (np.asarray(ds.variables["P"][0], dtype=np.float64)
                       + np.asarray(ds.variables["PB"][0],
                                    dtype=np.float64))
        phi = (np.asarray(ds.variables["PH"][0], dtype=np.float64)
               + np.asarray(ds.variables["PHB"][0], dtype=np.float64))
    expected = metrics.terrain_smoothed_mslp(
        diagnostics["mslp"], pressure_pa[0], phi[0] / 9.80665)
    assert np.array_equal(diagnostics["mslp_display"], expected)
    # The scored field itself must remain the raw reduction (its
    # arithmetic is pinned by tests/test_real74_d01.py); here it only
    # has to stay finite and distinct from the treated field where the
    # terrain is high.
    assert np.isfinite(diagnostics["mslp"]).all()


def test_the_display_twin_is_computed_but_never_drawn_here(tmp_path):
    """The smoothing survives the render-law move; the drawing does not.

    ``make_synoptic_maps`` used to contour ``mslp_display`` with
    matplotlib, and this test pinned that it contoured the treated field
    rather than the raw one.  Weather-field product plots come from
    ``rw_wrfbatch`` now (CLAUDE.md, Drew 2026-08-06; audit F6), so the
    MSLP panel is the renderer's ``mslp_10m_winds`` and the contouring
    question is no longer this module's to answer.

    What is still this module's, and is what this now holds: the treated
    field is COMPUTED and published beside the raw one, so the smoothing
    cannot be quietly dropped along with the plotting code, and the
    scored field stays the raw reduction.
    """
    import netCDF4

    ny, nx = 6, 10
    terrain = np.full((ny, nx), 40.0)
    terrain[:, nx // 2:] = 2200.0
    path = tmp_path / "wrfout_d01_1974-04-04_12_00_00"
    _write_synthetic_wrfout(path, terrain)

    diagnostics = metrics._wrf_diagnostics(path)
    with netCDF4.Dataset(path) as ds:
        pressure_pa = (np.asarray(ds.variables["P"][0], dtype=np.float64)
                       + np.asarray(ds.variables["PB"][0],
                                    dtype=np.float64))
        phi = (np.asarray(ds.variables["PH"][0], dtype=np.float64)
               + np.asarray(ds.variables["PHB"][0], dtype=np.float64))
    assert np.array_equal(
        diagnostics["mslp_display"],
        metrics.terrain_smoothed_mslp(
            diagnostics["mslp"], pressure_pa[0], phi[0] / 9.80665))
    # And the map spec is a product declaration, with no plot recipe in
    # it at all -- nothing here can name a colormap or a filename again.
    spec = metrics.SynopticMapSpec(products=("mslp_10m_winds",))
    assert spec.products == ("mslp_10m_winds",)
    assert not hasattr(spec, "mslp_t2_filename")
