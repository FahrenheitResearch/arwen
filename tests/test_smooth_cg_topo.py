"""WRF v4.7.1 smooth_cg_topo: d01's terrain blended toward the source's.

The operator is WRF's own ``blend_terrain`` (dyn_em/nest_init_utils.F),
cut verbatim into the oracle (tools/wrf_topo_radiation_v471_oracle) and
run the way real.exe calls it on d01, with the source terrain (toposoil)
as the coarse field.  The port is held bit for bit to it.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest.cg_topo import (RootTerrainBlend, apply_smooth_cg_topo,
                                  blend_root_terrain)

FIXTURE = (Path(__file__).parent / "fixtures" / "wrf_topo_radiation_v471"
           / "topo_radiation_v471.npz")


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _cases(fx):
    return sorted({"/".join(k.split("/")[:3]) for k in fx.files
                   if k.startswith("blend/") and "/sbw" in k})


def test_blend_matches_wrf_blend_terrain_bit_for_bit(fx):
    cases = _cases(fx)
    assert len(cases) == 8
    for key in cases:
        name = key.split("/")[1]
        sbw, width = (int(part[3:]) if part.startswith("sbw")
                      else int(part[2:])
                      for part in key.split("/")[2].split("_"))
        ht = fx[f"blend/{name}/ht"]
        # gpuwm's statics are float64; the blend reads them as WRF's REAL.
        out = blend_root_terrain(
            ht.astype(np.float64), fx[f"blend/{name}/toposoil"],
            spec_bdy_width=sbw, blend_width=width)
        np.testing.assert_array_equal(out.astype(np.float32),
                                      fx[f"{key}/cr/ht"], err_msg=key)
        np.testing.assert_array_equal(fx[f"{key}/cr/ht"],
                                      fx[f"{key}/glibc/ht"], err_msg=key)


def test_blend_leaves_the_interior_and_keeps_float64(fx):
    ht = fx["blend/alps_geogrid/ht"].astype(np.float64) + 0.25
    coarse = fx["blend/alps_geogrid/toposoil"]
    out = blend_root_terrain(ht, coarse, spec_bdy_width=5, blend_width=5)
    assert out.dtype == np.float64
    interior = (slice(10, -10), slice(10, -10))
    np.testing.assert_array_equal(out[interior], ht[interior])
    np.testing.assert_array_equal(out[:5], coarse[:5].astype(np.float64))
    assert (out[5:10, 10:-10] != ht[5:10, 10:-10]).any()


def test_off_is_the_identity_and_on_needs_source_terrain():
    static = {"HGT_M": np.ones((20, 24))}
    off = SimpleNamespace(smooth_cg_topo=False, spec_bdy_width=5,
                          blend_width=5)
    assert apply_smooth_cg_topo(off, static, source_orography=None,
                                route="test") is static
    on = SimpleNamespace(smooth_cg_topo=True, spec_bdy_width=5,
                         blend_width=5)
    with pytest.raises(ValueError, match="SOILHGT is required"):
        apply_smooth_cg_topo(on, static, source_orography=None,
                             route="test")


def test_route_blend_runs_once_and_reuses_the_first_terrain():
    """WRF blends at real.exe's first time and reuses ht_smooth after."""
    static = {"HGT_M": np.full((40, 44), 1000.0)}
    exp = SimpleNamespace(smooth_cg_topo=True, spec_bdy_width=5,
                          blend_width=5)
    blend = RootTerrainBlend(exp, static, route="test")
    blend.before_initialize(np.full((40, 44), 200.0, np.float32))
    first = static["HGT_M"].copy()
    assert first[0, 0] == 200.0 and first[20, 20] == 1000.0
    blend.before_initialize(np.full((40, 44), 900.0, np.float32))
    np.testing.assert_array_equal(static["HGT_M"], first)
