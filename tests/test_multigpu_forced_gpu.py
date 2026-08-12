"""The FORCED decomposition rung, as a pytest door onto the real gate.

One test, deliberately: ``tilestream.multigpu.gate_forced`` is the verdict
machine (specified case, 1/2/4 ranks vs the resident digest, poison-seam and
scaled-edge controls) and this file only opens the door the battery walks
through.  The full-size rung stays reachable as
``python -m tilestream.multigpu forced``.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_forced_gate_small():
    from tilestream import multigpu as mg

    ok, results = mg.gate_forced(160, 132, steps=4, verbose=True)
    assert results["runs"], "gate ran no rank geometries"
    for key, row in results["runs"].items():
        assert row["match"], (key, row.get("diff"))
    assert results["controls"]["poison_seam_matches"]
    assert results["controls"]["scaled_edge_differs"]
    assert ok


def test_scatter_geography_imposes_domain_flags():
    """realcase_mg.scatter_geography must impose the DOMAIN's
    has_msf/rotational alongside the arrays it installs.

    The rank buffers are built on ``harness.neutral_geography``, whose
    docstring makes the imposition load-bearing by design: the neutral build
    derives both flags False, and writing real msf/f arrays into the buffer
    is exactly the bypass ``set_map_coriolis`` warns about -- the arrays
    change, the flags do not, and every msf-weighted dycore path plus the
    Coriolis+curvature kernel silently stay off.  That was ~0.3% everywhere
    in the decomposed real-case tendencies, bit-reproducible and NaN-free.
    """
    from types import SimpleNamespace

    import cupy as cp

    from tilestream import driver, harness, multigpu as mg
    from tilestream import realcase_mg as rmg

    cfg = mg.forced_config(64, 48, 3)
    state = harness.make_state(cfg, geography=harness.neutral_geography(cfg))
    assert state.has_msf is False and state.rotational is False

    inv = driver.geography_inventory(state)
    geo_home = {k: np.array(cp.asnumpy(v) if isinstance(v, cp.ndarray)
                            else v, dtype=np.float64) for k, v in inv.items()}
    specs = mg.plan_split(cfg.nx, cfg.ny, mg.forced_halo(cfg),
                          gx=1, gy=1, periodic=False)
    dom = SimpleNamespace(devices=[0], specs=specs, states=[state],
                          nz=cfg.nz, ny=cfg.ny, nx=cfg.nx)

    # A neutral domain imposes neutral flags -- the imposition follows the
    # DOMAIN's truth, it is not a blind True.
    assert rmg.scatter_geography(dom, geo_home) > 0
    assert state.has_msf is False and state.rotational is False

    # A rotational domain with real map factors must flip both flags ON.
    geo_home["setup/msft"][...] = 1.003
    geo_home["setup/msfu"][...] = 1.003
    geo_home["setup/msfv"][...] = 1.003
    geo_home["setup/f"][...] = 1.0e-4
    assert rmg.scatter_geography(dom, geo_home) > 0
    assert state.has_msf is True and state.rotational is True
    assert float(cp.asnumpy(state.msft).min()) == pytest.approx(1.003)
