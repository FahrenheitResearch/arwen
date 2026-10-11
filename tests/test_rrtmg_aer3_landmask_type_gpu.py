"""aer_opt = 3 land/water aerosol type on the device.

The CPU twin, the tables and the configuration are held by
tests/test_rrtmg_aer3_landmask_type.py.  Here: the CUDA kernel
``rrtmg_aer3_typed_sw_optics`` against the NOAA MPAS fork's Fortran word
for word in the batched SW engine's (column, band, layer) layout, the
device xland-to-type mapping, the typed kernel on an all-land deck against
the untouched rural kernel, and the legacy adapter's restart identity.
Run once with GPUWM_WRF_EXACT=1 and once without: the kernel spells every
operation as an explicit single-rounded intrinsic, so both read 0 words.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from gpuwm.core import rrtmg_aerosol_optics as ao
from test_rrtmg_aer3_landmask_type import INPUTS, bits_equal, oracle  # noqa: F401

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu


def _device_inputs(o):
    return [cp.asarray(o[f"in/{k}"]) for k in INPUTS]


def test_device_type_is_the_noaa_drivers(oracle):
    got = ao.aer_type_from_xland_device(cp.asarray(oracle["in/xland"]))
    assert got.dtype == cp.int32
    assert np.array_equal(cp.asnumpy(got), oracle["out/aer_type"])
    with pytest.raises(ValueError, match="assigns no aerosol type"):
        ao.aer_type_from_xland_device(cp.asarray(np.array([1, 0, 2], np.float32)))


def test_kernel_matches_the_noaa_mpas_fork_word_for_word(oracle):
    o = oracle
    dev = _device_inputs(o)
    nz = dev[0].shape[1]
    aer_type = ao.aer_type_from_xland_device(cp.asarray(o["in/xland"]))
    ztaua, zasya, zomga, taod = ao.aer3_sw_optics_device(
        *dev, nz + 1, aer_type=aer_type)
    want = ao.sw_engine_layout(o["out/tauaer"], o["out/ssaaer"],
                               o["out/asyaer"], nz + 1)
    words = 0
    for name, got, w in zip(("ztaua", "zasya", "zomga"),
                            (ztaua, zasya, zomga), want):
        words += bits_equal(name, cp.asnumpy(got), w)
    words += bits_equal("taod5503d", cp.asnumpy(taod), o["out/taod5503d"])
    wide = ao.aer3_sw_optics_device(*dev, nz + 3, aer_type=aer_type)
    bits_equal("wide tau model layers", cp.asnumpy(wide[0])[:, :, :nz],
               want[0][:, :, :nz])
    assert np.all(cp.asnumpy(wide[0])[:, :, nz:] == 0)
    assert np.all(cp.asnumpy(wide[2])[:, :, nz:] == 1)
    print(f"aer3 landmask kernel vs NOAA MPAS fork: {words} words, max ULP 0")


def test_typed_kernel_on_land_is_the_rural_kernel(oracle):
    o = oracle
    dev = _device_inputs(o)
    nc, nz = dev[0].shape
    rural = ao.aer3_sw_optics_device(*dev, nz + 1)
    typed = ao.aer3_sw_optics_device(
        *dev, nz + 1, aer_type=cp.ones(nc, cp.int32))
    for name, got, want in zip(("ztaua", "zasya", "zomga", "taod"), typed, rural):
        bits_equal(f"all-land {name}", cp.asnumpy(got), cp.asnumpy(want))


def test_device_refusals(oracle):
    o = oracle
    dev = _device_inputs(o)
    nc, nz = dev[0].shape
    with pytest.raises(ValueError, match="int32 device array"):
        ao.aer3_sw_optics_device(*dev, nz + 1, aer_type=cp.ones(nc, cp.int64))
    with pytest.raises(ValueError, match="1 \\(rural\\) and 3"):
        ao.aer3_sw_optics_device(*dev, nz + 1,
                                 aer_type=cp.full(nc, 2, cp.int32))
    with pytest.raises(ValueError, match="no smoke coupling"):
        ao.aer3_sw_optics_device(*dev, nz + 1, aer_type=cp.ones(nc, cp.int32),
                                 smoke_aod=cp.zeros((nc, nz), cp.float32),
                                 smoke_feedback=True)


def test_the_adapter_identity_binds_the_maritime_rows_only_when_selected():
    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
    lat = np.full((1, 2), 40.0, np.float32)
    lon = np.full((1, 2), -100.0, np.float32)
    start = datetime(2026, 10, 3, 12)
    rural = RRTMGLegacyRadiation(start, lat, lon, aer_opt=3).restart_identity()
    land = RRTMGLegacyRadiation(start, lat, lon, aer_opt=3,
                                aer3_type_source="landmask").restart_identity()
    assert "aer3_type_source" not in rural
    assert "aer3_maritime_tables_sha256" not in rural
    assert land["aer3_type_source"] == "landmask"
    assert land["aer3_maritime_tables_sha256"] == ao.AER3_MARITIME_TABLES_SHA256
    assert {k: v for k, v in land.items()
            if k not in ("aer3_type_source", "aer3_maritime_tables_sha256")} == rural


def test_the_actual_shortwave_adapter_moves_only_water_columns():
    """Through the legacy adapter the shortwave call reads: the land/water
    type leaves every land column's surface shortwave and heating rate the
    same words as the rural optics and changes every sunlit water column."""
    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
    from test_rrtmg_legacy_wiring import _bundle, _env_from, _call, profile

    start = datetime(2026, 10, 2, 21)
    ny, nx = 2, 3
    lat = np.full((ny, nx), 40.0, np.float32) + np.float32(0.02) * np.arange(ny)[:, None]
    lon = np.full((ny, nx), -100.0, np.float32) + np.float32(0.02) * np.arange(nx)[None, :]
    bundle = _bundle(profile.__wrapped__(), ny * nx, seed=20261008)
    env = _env_from(bundle, np.arange(ny * nx), ny, nx, lat.ravel(), lon.ravel())
    env.state.nwfa = cp.full_like(env.state.qv, cp.float32(3.e9))
    env.state.nifa = cp.full_like(env.state.qv, cp.float32(1.e6))
    env.state.elapsed_seconds = 0.0
    env.fields["glw"] = cp.full((ny, nx), cp.float32(300.))
    xland = np.array([[1., 2., 1.], [2., 1., 2.]], np.float32)
    env.fields["xland"] = cp.asarray(xland)
    env.cfg.mp_physics = 28
    env.cfg.aer_opt = 3
    env.cfg.o3input = 0
    env.cfg.swint_opt = 0
    env.cfg.rrtmg_smoke_manifest = ""

    def run(source):
        env.cfg.rrtmg_aer3_type_source = source
        adapter = RRTMGLegacyRadiation(
            start, lat, lon, p_top=env.p_top, o3input=0, longwave=False,
            shortwave=True, aer_opt=3, aer3_type_source=source)
        result = _call(adapter, env)
        return (cp.asnumpy(result.swdown),
                cp.asnumpy(result.rthratensw).reshape(-1, ny, nx))

    rural_sw, rural_hr = run("rural")
    land_sw, land_hr = run("landmask")
    land = xland == 1
    assert (rural_sw > 0).all(), "the numerical fixture must be sunlit"
    np.testing.assert_array_equal(land_sw[land].view(np.uint32),
                                  rural_sw[land].view(np.uint32))
    np.testing.assert_array_equal(land_hr[:, land].view(np.uint32),
                                  rural_hr[:, land].view(np.uint32))
    assert np.isfinite(land_sw).all() and np.isfinite(land_hr).all()
    assert (np.abs(land_sw[~land] - rural_sw[~land]) > 0.01).all(), (
        land_sw[~land], rural_sw[~land])
    print("adapter swdown rural", rural_sw.tolist(), "landmask", land_sw.tolist())
