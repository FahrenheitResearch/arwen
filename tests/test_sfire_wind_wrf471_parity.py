"""Full wind driver against a byte-extracted, compiled WRF routine."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from conftest import requires_gpu


def _load(name):
    root = Path(__file__).parents[1] / "tools/sfire_wrf471_oracle/fixtures/wind"
    receipt = json.loads((root / "receipt.json").read_text())
    path = root / (name + ".npz")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["cases"][name]["sha256"]
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


@requires_gpu
@pytest.mark.parametrize("case", range(1, 13))
@pytest.mark.parametrize("layout", ["native", "tiled"])
def test_full_wind_driver_matches_original_wrf(case, layout):
    import cupy as cp
    from gpuwm.core.sfire_wind import interpolate_atm2fire
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f = _load(f"wind/{layout}_{case}")
    args = dict(fire_wind_height=float(f["fire_wind_height"]),
                domain=tuple(f["domain"]), fine_domain=tuple(f["fine_domain"]),
                fire_lsm_zcoupling=bool(f["fire_lsm_zcoupling"]),
                fire_lsm_zcoupling_ref=float(f["fire_lsm_zcoupling_ref"]),
                u_frame=0.15, v_frame=-0.12)
    if layout == "tiled":
        args.update(tiles=tuple(map(tuple, f["tiles"])), fine_tiles=tuple(map(tuple, f["fine_tiles"])))
    got = interpolate_atm2fire(f["u"][:-1], f["v"][:-1], *(f[k] for k in ("ph", "phb", "z0", "zs", "z0f")),
                               int(f["sr_x"]), int(f["sr_y"]), **args)
    xl, xh, yl, yh = args["domain"]
    fl, fh, fb, ft = args["fine_domain"]
    for name in ("uf", "vf"):
        assert_bit_exact(cp.asnumpy(got[name][fb:ft+1, fl:fh+1]), f[name][fb:ft+1, fl:fh+1], name)
    for name in ("uah", "vah"):
        assert_bit_exact(cp.asnumpy(got[name][yl:yh+2, xl:xh+2]), f[name], name)


@requires_gpu
@pytest.mark.parametrize("case", range(1, 10))
@pytest.mark.parametrize("staggered", [False, True])
def test_compact_native_adapter_matches_original_wrf(case, staggered):
    import cupy as cp
    from gpuwm.core.sfire_wind import interpolate_native_atm2fire
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f = _load(f"wind/compact_{case}")
    xl, xh, yl, yh = map(int, f["domain"])
    got = interpolate_native_atm2fire(
        f["u"][:-1, yl:yh, xl:xh+1], f["v"][:-1, yl:yh+1, xl:xh],
        f["ph"][:, yl:yh, xl:xh], f["phb"][:, yl, xl] if case >= 7 else f["phb"][:, yl:yh, xl:xh],
        f["z0"][yl:yh, xl:xh], f["zs"][yl:yh, xl:xh], f["z0f"],
        int(f["sr_x"]), int(f["sr_y"]), fire_wind_height=float(f["fire_wind_height"]),
        return_staggered_diagnostics=staggered,
        domain_includes_terminal_face=True,
        fire_lsm_zcoupling=bool(f["fire_lsm_zcoupling"]),
        fire_lsm_zcoupling_ref=float(f["fire_lsm_zcoupling_ref"]))
    fl, fh, fb, ft = map(int, f["fine_domain"])
    for name in ("uf", "vf"):
        assert_bit_exact(cp.asnumpy(got[name][fb:ft+1, fl:fh+1]), f[name][fb:ft+1, fl:fh+1], name)
    for name in ("uah", "vah"):
        assert_bit_exact(cp.asnumpy(got[name]), f[name + "_staggered"] if staggered else f[name], name)


@requires_gpu
@pytest.mark.parametrize("case", range(1, 10))
@pytest.mark.parametrize("staggered", [False, True])
def test_coupled_driver_mass_bounds_match_original_wrf(case, staggered):
    import cupy as cp
    from gpuwm.core.sfire_wind import interpolate_native_atm2fire
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f = _load(f"wind/drivercompact_{case}")
    xl, xh, yl, yh = map(int, f["domain"])
    got = interpolate_native_atm2fire(
        f["u"][:-1, yl:yh+1, xl:xh+2], f["v"][:-1, yl:yh+2, xl:xh+1],
        f["ph"][:, yl:yh+1, xl:xh+1], f["phb"][:, yl, xl] if case >= 7 else f["phb"][:, yl:yh+1, xl:xh+1],
        f["z0"][yl:yh+1, xl:xh+1], f["zs"][yl:yh+1, xl:xh+1], f["z0f"],
        int(f["sr_x"]), int(f["sr_y"]), fire_wind_height=float(f["fire_wind_height"]),
        return_staggered_diagnostics=staggered,
        fire_lsm_zcoupling=bool(f["fire_lsm_zcoupling"]),
        fire_lsm_zcoupling_ref=float(f["fire_lsm_zcoupling_ref"]))
    fl, fh, fb, ft = map(int, f["fine_domain"])
    for name in ("uf", "vf"):
        assert_bit_exact(cp.asnumpy(got[name][fb:ft+1, fl:fh+1]), f[name][fb:ft+1, fl:fh+1], name)
    for name in ("uah", "vah"):
        assert_bit_exact(cp.asnumpy(got[name]), f[name + "_staggered"] if staggered else f[name], name)
