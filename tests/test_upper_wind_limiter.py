"""The fork's descending saved-wind limiter, graded against its Fortran block."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.config import RunConfig, validate_run_config


def test_default_keeps_the_existing_acoustic_launches():
    from gpuwm.core.acoustic import prepare_upper_wind_limiter
    cfg = RunConfig(nx=7, ny=5, nz=6, dx=3000., dy=3000.,
                    ztop=20000., dt=20., run_seconds=60.)
    assert cfg.upper_wind_limiter_form == "wrf_461"
    assert prepare_upper_wind_limiter(None, cfg, 5.0) is None
    cfg = replace(cfg, upper_wind_limiter_form="noaa_wrf39", damp_opt=0)
    assert prepare_upper_wind_limiter(None, cfg, 5.0) is None
    cfg = replace(cfg, upper_wind_limiter_form="unknown")
    with pytest.raises(ValueError, match="upper_wind_limiter_form"):
        validate_run_config(cfg)


@requires_gpu
@pytest.mark.gpu
def test_fork_saved_wind_words_match_the_compiled_source_block():
    import cupy as cp
    from gpuwm.core.acoustic import prepare_upper_wind_limiter
    path = Path(__file__).parent / "data" / "upper_wind_fork.npz"
    with np.load(path) as f:
        for c in range(int(f["cases"])):
            p = str(c) + "_"
            cfg = RunConfig(nx=7, ny=5, nz=6, dx=3000., dy=3000.,
                            ztop=20000., dt=20., run_seconds=60., damp_opt=3,
                            zdamp=float(f[p + "zdamp"]),
                            specified=bool(f[p + "zone"]),
                            upper_wind_limiter_form="noaa_wrf39")
            state = SimpleNamespace(**{n: cp.asarray(f[p + n])
                                      for n in ("u", "v", "php", "phb")})
            state.thb = state.phb
            launch = prepare_upper_wind_limiter(state, cfg, float(f[p + "dts"]))
            for _ in range(int(f[p + "substeps"])):
                launch()
            for n in ("u", "v"):
                np.testing.assert_array_equal(state.__dict__[n].get().view(np.uint32),
                    f[p + n + "_out"].view(np.uint32), err_msg=p+n)


def test_fork_import_also_selects_the_saved_wind_limiter(tmp_path):
    from test_diff6_fork_form import _import
    doc, _ = _import(tmp_path, dynamics=" diff_6th_factor2 = 0.04, 0.04,\n")
    assert doc["shared"]["upper_wind_limiter_form"] == "noaa_wrf39"
    stock, _ = _import(tmp_path)
    assert "upper_wind_limiter_form" not in stock["shared"]


@requires_gpu
@pytest.mark.gpu
def test_scalar_reference_fluxes_read_the_limited_saved_winds():
    import cupy as cp
    from gpuwm.core.dycore import stage_fluxes, refresh_saved_wind_fluxes
    from gpuwm.verify.npref import random_acoustic_state
    state, cfg = random_acoustic_state(seed=83, ny=6, nx=12)
    ru, rv, ww = stage_fluxes(state, cfg)
    old_ru, old_rv, old_ww = ru.copy(), rv.copy(), ww.copy()
    # A known saved-wind change at the same point the limiter writes.
    state.u *= cp.float32(0.5)
    state.v *= cp.float32(0.5)
    refresh_saved_wind_fluxes(state, cfg, ru, rv)
    np.testing.assert_array_equal(ru.get(), (old_ru*cp.float32(0.5)).get())
    np.testing.assert_array_equal(rv.get(), (old_rv*cp.float32(0.5)).get())
    np.testing.assert_array_equal(ww.get(), old_ww.get())
