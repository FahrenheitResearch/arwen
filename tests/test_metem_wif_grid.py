"""The met_em route hands initialize_real the mass-point lat/lon it holds.

prepare_metem_run computes ``grid.latlon_mass()`` and checks it against the
met_em file, then called initialize_real with no ``grid=``.  An mp=28 met_em
run on the WIF climatology (aer_init_opt = wif_input_opt = 1) whose met_em
lacks QNWFA/QNIFA builds nwfa/nifa from those coordinates, and so refused
"could not derive the model mass-point latitudes/longitudes" while holding
them.  The HRRR route's boundary strips got the same fix in e62827329.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from test_preparation_door_order import card  # noqa: F401  (fixture)


class _Reached(Exception):
    """The door reached initialize_real; the test stops it there."""


def test_the_met_em_door_passes_its_mass_point_coordinates(card, monkeypatch,  # noqa: F811
                                                           tmp_path):
    import gpuwm.ingest.metem as metem
    import gpuwm.ingest.real as real
    from gpuwm import metem_door, metem_forecast
    from gpuwm.static.projection import grids_from_projection_config
    from test_metem_memory_advisory import _case

    run = _case(tmp_path, monkeypatch, device_low=False, host_low=False)
    exp = run.experiment
    monkeypatch.setattr(metem_forecast, "read_met_em_terrain",
                        lambda path: np.zeros((exp.root.run.ny,
                                               exp.root.run.nx)))
    monkeypatch.setattr(metem_forecast, "adapt_experiment_vertical",
                        lambda experiment, *_a, **_k: (experiment, None))
    grids = {int(domain.grid_id): grid for domain, grid in zip(
        exp.domains, grids_from_projection_config(exp))}

    def read_met_em(path):
        grid_id = next(gid for gid, files in run.paths.items()
                       if path in files)
        lat, lon = grids[grid_id].latlon_mass()
        return SimpleNamespace(
            path=path, snapshot=None, terrain=None, source_orography=None,
            statics={"XLAT_M": lat, "XLONG_M": lon})

    monkeypatch.setattr(metem, "read_met_em", read_met_em)
    monkeypatch.setattr(metem, "met_em_series_identity", lambda case: {})
    monkeypatch.setattr(metem_door, "metgrid_initialization_controls",
                        lambda case, run, cfg=None: {})
    calls = []

    def initialize_real(snapshot, cfg, coord, terrain, **kwargs):
        calls.append(kwargs)
        raise _Reached

    monkeypatch.setattr(real, "initialize_real", initialize_real)
    with pytest.raises(_Reached):
        # The CPU stand-in backend: stage 1 runs with GPUWM_NO_LOCAL_GPU.
        metem_forecast.prepare_metem_run(run, tmp_path / "prepared",
                                         preprocess_backend="cpu")
    (kwargs,) = calls
    lat, lon = grids[int(exp.domains[0].grid_id)].latlon_mass()
    np.testing.assert_array_equal(kwargs["grid"]["XLAT_M"], lat)
    np.testing.assert_array_equal(kwargs["grid"]["XLONG_M"], lon)
    # And the initializer's WIF door reads exactly those coordinates from it.
    derived = real._wif_grid_latlon_from(kwargs["grid"], None)
    np.testing.assert_array_equal(derived[0], lat)
    np.testing.assert_array_equal(derived[1], lon)


def test_a_caller_with_no_coordinates_is_still_refused_by_name():
    """The refusal stays for a configuration that truly lacks them."""
    import gpuwm.ingest.real as real

    assert real._wif_grid_latlon_from(None, None) is None
    assert real._wif_grid_latlon_from(None, SimpleNamespace()) is None
    text = open(real.__file__, encoding="utf-8").read()
    assert "the model mass-point latitudes/longitudes (pass" in text
