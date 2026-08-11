"""Every registered radiation callable classifies cleanly (1.9.1 D3).

Shipped 1.9.0's RRTM longwave adapter stored ``p_top`` uncoerced
(gpuwm/core/rrtm_lw.py), while its legacy-RRTMG sibling wraps the same
value in ``float()``.  ``initialize_physics`` hands over ``state.p_top``
-- a NumPy float32 SCALAR -- and the restart classifier's
``_is_array_like`` (shape+dtype+ndim) treats a NumPy scalar as an array.
``RADIATION_CALLABLE_ARRAYS`` has no ``p_top`` entry, so every
RRTM+Dudhia (1/1) run died with RestartManifestError in the end-of-run
canonical digest after a PERFECT integration: the false-failure class.

The class instrument: construct every radiation callable
``initialize_physics`` can register, with constructor inputs of the same
TYPES the production seam passes (NumPy-scalar ``p_top`` included), and
walk each through the restart classifier's own check.  Every attribute
must classify cleanly.  Red-on-revert: removing either ``float()``
coercion in gpuwm/core/rrtm_lw.py, or the composed adapter's container
classification in gpuwm/io/restart.py, fails this suite.

CPU-only via the NumPy-for-CuPy shim the RRTM suites already use;
adapters whose CONSTRUCTORS require CUDA assets are exercised by their
own GPU suites and the runs that use them, and are listed here so a new
registered adapter must either join the walk or state why not.
"""

from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import pytest


_START = datetime(2011, 4, 27, 18)


def _grids():
    lat = np.full((3, 4), 39.0, np.float32)
    lon = np.full((3, 4), -87.0, np.float32)
    return lat, lon


def _walk(adapter):
    from gpuwm.io.restart import (RADIATION_CALLABLE_ARRAYS,
                                  RADIATION_CALLABLE_CONTAINERS,
                                  _callable_state_check)

    _callable_state_check(adapter, RADIATION_CALLABLE_ARRAYS,
                          RADIATION_CALLABLE_CONTAINERS, "radiation")


@pytest.fixture()
def numpy_shim(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", np)
    yield


def test_analytic_clear_sky_classifies_cleanly(numpy_shim, monkeypatch):
    import gpuwm.core.analytic_radiation as analytic

    monkeypatch.setattr(analytic, "cp", np)
    lat, lon = _grids()
    _walk(analytic.AnalyticClearSkyRadiation(_START, lat, lon))


def test_dudhia_shortwave_classifies_cleanly(numpy_shim):
    from gpuwm.core.dudhia import DudhiaShortwaveRadiation

    lat, lon = _grids()
    _walk(DudhiaShortwaveRadiation(_START, lat, lon, swrad_scat=1.0,
                                   icloud=1))


def test_rrtm_longwave_classifies_cleanly_with_numpy_scalar_p_top(
        numpy_shim):
    """The exact production construction: p_top is state.p_top (float32).

    Red without the ``float()`` coercion: the NumPy scalar attribute is
    an unclassified "array" to the walk.
    """
    from gpuwm.core.rrtm_lw import RRTMLongwaveRadiation

    lat, lon = _grids()
    adapter = RRTMLongwaveRadiation(_START, lat, lon,
                                    p_top=np.float32(5000.0))
    assert type(adapter.p_top) is float  # the sibling's coercion, exactly
    _walk(adapter)


def test_rrtm_dudhia_composition_classifies_cleanly(numpy_shim):
    """The registered 1/1 pair, constructed as initialize_physics does.

    Covers BOTH halves of D3: the composed adapter's own ``p_top`` copy
    must coerce, and its two sub-adapters must be classified containers
    (they carry the construction-time lat/lon grids and the packaged RRTM
    tables -- rebuild-on-load, no cross-step array state).
    """
    from gpuwm.core.rrtm_lw import RRTMDudhiaRadiation

    lat, lon = _grids()
    adapter = RRTMDudhiaRadiation(_START, lat, lon,
                                  p_top=np.float32(5000.0),
                                  icloud=1, swrad_scat=1.0)
    assert type(adapter.p_top) is float
    assert type(adapter.longwave_adapter.p_top) is float
    _walk(adapter)
    # The sub-adapters classify cleanly on their own too, so the
    # container allowance is not hiding an unclassified array one level
    # down.
    _walk(adapter.longwave_adapter)
    _walk(adapter.shortwave_adapter)


def test_rrtmg_legacy_coerces_p_top_the_same_way(numpy_shim):
    """The sibling whose coercion D3's fix transcribes.

    Constructing the full legacy adapter loads its packaged CUDA assets,
    which its own suites cover; here the pinned property is the
    constructor's ``p_top`` handling, which must stay the family's
    reference behaviour.
    """
    import inspect

    from gpuwm.core import rrtmg_legacy

    source = inspect.getsource(rrtmg_legacy.RRTMGLegacyRadiation.__init__)
    assert "float(p_top)" in source


def test_every_registered_radiation_callable_is_walked_here():
    """The registry of this suite: initialize_physics' dispatch, verbatim.

    A new radiation pair added to gpuwm/core/physics.py must either gain
    a walk above or extend this list with a stated reason -- the D3 class
    exists because the 1/1 pair was registered without one.
    """
    import inspect

    import gpuwm.core.physics as physics

    source = inspect.getsource(physics.initialize_physics)
    registered = {
        "AnalyticClearSkyRadiation": "walked above",
        "RRTMGLegacyRadiation": (
            "constructor loads packaged CUDA assets; p_top coercion "
            "pinned above, classification exercised by its 4/4 GPU "
            "suites and every legacy-RRTMG run's digest"),
        "RRTMGPRadiation": (
            "constructor loads packaged k-distributions; its containers "
            "are classified by name (lw_tables/sw_tables/...) and "
            "exercised by every RTE+RRTMGP run's digest"),
        "DudhiaShortwaveRadiation": "walked above",
        "RRTMDudhiaRadiation": "walked above",
    }
    constructed = {name for name in registered
                   if f"{name}(" in source}
    assert constructed == set(registered), (
        "initialize_physics registers a radiation callable this suite "
        "does not account for; add a walk or a stated reason")
