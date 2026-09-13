"""Device staging of an observation batch's reach window, on a real card.

THE DEFECT.  ``gpuwm.da.radar_assimilation.assimilate_radar_grid`` stages
every batch onto the device by rebuilding
:class:`~gpuwm.da.letkf.GriddedObs` field by field, and the rebuild dropped
``window``.  ``gpuwm.da.obs_radar`` builds velocity batches from a v2
reach-window document with window-sized arrays and the window that names
them, so on the accelerator those arrays reached the filter declaring no
window at all.  ``gpuwm.da.letkf`` sizes every batch array against the
declaration it is given, so it raised ``LetkfError`` on a shape it had
itself produced, and a windowed document could not be assimilated on a
card.  The CPU arm never rebuilds a batch and was never affected, which is
exactly why a CPU leg cannot ask this question.

``tests/test_local_da_integration.py::test_accelerator_staging_preserves_observation_window``
pins the same contract on stage 1 with a stand-in device namespace whose
``asarray`` is numpy's.  A stand-in cannot show that the real transfer
preserves the declaration, and the transfer is the step that dropped it.

The two halves here are the two halves of the defect: that the staging
carries ``window`` across a real cupy transfer, and that the filter on the
card needs that declaration, so dropping it is what raised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]

NZ, NY, NX = 2, 5, 5
MEMBERS = 3


def _grid():
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid

    return TargetGrid.from_projection(
        LambertGrid(ref_lat=35., ref_lon=-97., truelat1=33., truelat2=37.,
                    stand_lon=-97., dx=3000., dy=3000., e_we=6, e_sn=6),
        z_w=np.array([0., 1500., 4000.]), name='staging-fixture')


def _checkpoints(tmp_path, count=MEMBERS):
    out = {}
    for index in range(count):
        root = tmp_path / f'member_{index:03d}'
        root.mkdir()
        fields = dict(u=np.full((NZ, NY, NX + 1), 10. + index),
                      v=np.ones((NZ, NY + 1, NX)),
                      w=np.zeros((NZ + 1, NY, NX)),
                      thp=np.full((NZ, NY, NX), index * .1),
                      qv=np.full((NZ, NY, NX), .004),
                      p=np.broadcast_to(np.array([90000., 70000.])[:, None,
                                                                   None],
                                        (NZ, NY, NX)))
        path = root / 'gpuwmrst_d01_end.npz'
        np.savez(path, **{'state/' + k: v for k, v in fields.items()})
        out[index] = Path(path)
    return out


def _batch(window, shape, seed=7):
    from gpuwm.da.letkf import GriddedObs

    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, bool)
    mask.reshape(-1)[:4] = True
    simulated = rng.normal(1.5, .2, (MEMBERS, *shape))
    return GriddedObs(name='windowed-fixture',
                      values=np.where(mask, simulated.mean(axis=0) + .6,
                                      np.nan),
                      errors=np.full(shape, 0.5), simulated=simulated,
                      mask=mask, window=window)


def test_the_device_staging_carries_the_window_across_a_real_transfer(
        tmp_path):
    """Red before the rebuild carried ``window``: the staged batch said None.

    The batch is declared over the whole grid because that is the only
    shape ``extra_obs`` admits; the defect was the field being dropped, not
    the size it named, and this is the arm that runs a real cupy transfer.
    """
    import cupy as cp

    import gpuwm.da.radar_assimilation as owner
    from gpuwm.da.letkf import Localization

    shape = (NZ, NY, NX)
    staged = []

    def capture(prior, batches, geometry, cfg, diagnostics):
        for batch in batches:
            staged.append((batch.window,
                           type(batch.values).__module__.split('.')[0],
                           tuple(batch.values.shape)))
        return {name: cp.zeros_like(values)
                for name, values in prior.items()}

    config = owner.RadarAssimilationConfig(
        localization=Localization(6000., 3000.), rtps_alpha=0.,
        velocity=False, reflectivity=False, analysis_fields=('thp',),
        solve_device='cuda')
    owner.assimilate_radar_grid(
        _checkpoints(tmp_path), None, _grid(), config,
        extra_obs=[_batch((0, NY - 1, 0, NX - 1), shape)],
        extra_obs_provenance={'source': 'staging-fixture'},
        analysis_runner=capture)
    assert staged == [((0, NY - 1, 0, NX - 1), 'cupy', shape)], (
        'the batch handed to the filter must still declare the window it'
        ' was built with, and must be on the device')


def test_the_filter_on_the_card_needs_the_declaration_that_was_dropped():
    """What the dropped field cost: the same arrays, with and without it.

    A sub-window batch is what ``gpuwm.da.obs_radar`` builds from a v2
    reach-window document.  Declared, the card analyses it; stripped, which
    is what the staging used to hand over, the filter raises on the shape
    the window implies.
    """
    import cupy as cp

    from gpuwm.da.letkf import (GridGeometry, LetkfConfig, LetkfError,
                                Localization, analyze)
    from dataclasses import replace

    window = (1, 3, 1, 3)
    j0, j1, i0, i1 = window
    wshape = (NZ, j1 - j0 + 1, i1 - i0 + 1)
    rng = np.random.default_rng(11)
    grid = GridGeometry(dx_m=3000., dy_m=3000.,
                        heights_m=np.array([250., 1600.][:NZ]))
    prior = {'theta': cp.asarray(
        rng.standard_normal((MEMBERS, NZ, NY, NX)) + 300.)}
    batch = _batch(window, wshape)
    on_card = type(batch)(name=batch.name, values=cp.asarray(batch.values),
                          errors=cp.asarray(batch.errors),
                          simulated=cp.asarray(batch.simulated),
                          mask=cp.asarray(batch.mask), window=batch.window)
    config = LetkfConfig(localization=Localization(9000., 3000.),
                         analysis_fields=('theta',), rtps_alpha=0.)

    increments = analyze(prior, [on_card], grid, config)
    moved = float(cp.abs(increments['theta']).max())
    assert moved > 0., 'a declared sub-window batch moved nothing on the card'
    assert bool(cp.all(cp.isfinite(increments['theta'])))

    with pytest.raises(LetkfError) as raised:
        analyze(prior, [replace(on_card, window=None)], grid, config)
    assert 'shape' in str(raised.value) or 'window' in str(raised.value)
