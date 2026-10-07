"""The device LETKF reads a point batch without its dense H(x), and gives
the bytes it gives for the same observations stated densely."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpuwm.da.letkf import (GriddedObs, LetkfDiagnostics, PointSet,  # noqa: E402
                            point_batch)
from gpuwm.da.letkf_device import analyze_device  # noqa: E402
from test_letkf_device_gpu import geodesic_problem  # noqa: E402

pytestmark = pytest.mark.gpu


def test_point_batches_match_dense_batches_on_the_card():
    prior, obs, grid, cfg = geodesic_problem(batches=3)
    shape = next(iter(prior.values())).shape[1:]
    members = next(iter(prior.values())).shape[0]
    pointed = []
    for o in obs:
        if o.window is not None:
            pointed.append(o)
            continue
        mask = np.asarray(o.mask, bool)
        flat = np.flatnonzero(mask.reshape(-1))
        sim = np.asarray(o.simulated).reshape(members, -1)[:, flat]
        pointed.append(point_batch(
            o.name, shape,
            PointSet(flat_index=flat,
                     values=np.asarray(o.values).reshape(-1)[flat],
                     errors=np.asarray(o.errors).reshape(-1)[flat],
                     simulated=sim),
            localization=o.localization))
    dense = analyze_device(prior, obs, grid, cfg, LetkfDiagnostics())
    sparse = analyze_device(prior, pointed, grid, cfg, LetkfDiagnostics())
    for name in dense:
        assert np.array_equal(dense[name], sparse[name])
    assert any(getattr(o, "points", None) is not None for o in pointed)
    assert isinstance(obs[0], GriddedObs)
