"""RTE+RRTMGP edge ring and OLR level (lane rrtmgp-edge-olr, 2.9.1).

Breakage 1: under Morrison (the default suite) the specified-zone ring,
which the microphysics never runs on, kept its allocation-background radii
(2.5 um cloud, 5 um ice) and zero droplet number, and radiation read them:
on a default 3 h GFS forecast the ring's surface shortwave fell to a third
of its neighbour's under ice cloud and rose to nearly twice it under water
cloud.  Breakage 2: OLR was the upward longwave at the model top, below the
above-model layers RRTMGP appends, so it missed the column-top flux by up to
2.6 W m-2.
"""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu


def _column_case(cp, ny=7, nx=9, nz=30, ptop=5000.0):
    rng = np.random.default_rng(7)
    plev = np.broadcast_to(np.geomspace(100000.0, ptop, nz + 1)[:, None, None],
                           (nz + 1, ny, nx)).copy()
    play = np.sqrt(plev[:-1] * plev[1:])
    temperature = np.broadcast_to(
        np.linspace(295.0, 210.0, nz)[:, None, None], play.shape).copy()
    exner = (play / 1e5) ** (287.0 / 1004.0)
    qv = np.broadcast_to(np.geomspace(1e-2, 3e-6, nz)[:, None, None],
                         play.shape).copy()
    qc = np.zeros_like(play)
    qi = np.zeros_like(play)
    qs = np.zeros_like(play)
    qc[5:12] = 3e-4 * rng.random((7, ny, nx))
    qi[15:22] = 5e-5 * rng.random((7, ny, nx))
    qs[12:20] = 1e-4 * rng.random((8, ny, nx))
    f32 = lambda a: cp.asarray(a, dtype=cp.float32)  # noqa: E731
    atmosphere = {k: f32(v) for k, v in dict(
        pressure=play, p_interface=plev, temperature=temperature,
        theta=temperature / exner, exner=exner, qv=qv, qc=qc, qi=qi).items()}
    fields = {"tsk": cp.full((ny, nx), 295.0, cp.float32),
              "albedo": cp.full((ny, nx), 0.2, cp.float32),
              "emiss": cp.full((ny, nx), 0.98, cp.float32),
              "xland": cp.ones((ny, nx), cp.float32)}
    return atmosphere, fields, play, temperature, qi, qs, ptop, f32


@pytest.mark.gpu
@requires_gpu
def test_morrison_ring_radiates_its_own_moments_not_background_radii():
    cp = pytest.importorskip("cupy")
    from gpuwm.core import rrtmgp as rr

    atmosphere, fields, play, temperature, qi, qs, ptop, f32 = _column_case(cp)
    nz, ny, nx = play.shape
    rho = play / (287.15 * temperature)
    ring = np.zeros((ny, nx), bool)
    ring[0] = ring[-1] = True
    ring[:, 0] = ring[:, -1] = True
    zeros = np.zeros_like(play)
    ni = 1e5 * (qi > 0) / rho
    ns = 1e4 * (qs > 0) / rho
    rng = np.random.default_rng(3)
    # What the scheme leaves: its radii and fixed 250 cm-3 droplet number in
    # the tile, the allocation background on the ring it never runs on.
    written = dict(
        nc=np.where(ring, 0.0, 250e6 / rho),
        effc=np.where(ring, 2.5, 8.0 + 4.0 * rng.random(play.shape)),
        effi=np.where(ring, 5.0, 40.0) + zeros,
        effs=np.where(ring, 10.0, 90.0) + zeros)

    def run(*, specified, nc, updates, **radii):
        state = SimpleNamespace(
            elapsed_seconds=720.0, p_top=ptop, qc=atmosphere["qc"],
            qi=atmosphere["qi"], qs=f32(qs), qr=f32(zeros), nc=f32(nc),
            nr=f32(zeros), ni=f32(ni), ns=f32(ns),
            effr=f32(zeros + 25.0),
            physics=SimpleNamespace(microphysics_updates=updates),
            **{k: f32(v) for k, v in radii.items()})
        cfg = SimpleNamespace(mp_physics=10, dt=60.0, radt=12.0,
                              radt_minutes=12.0, specified=specified,
                              spec_zone=1)
        radiation = rr.RRTMGPRadiation(
            datetime(2023, 6, 21, 18), cp.full((ny, nx), 35.0),
            cp.full((ny, nx), -90.0), column_chunk=16)
        result = radiation(atmosphere=atmosphere, fields=fields,
                           state=state, cfg=cfg)
        return {n: cp.asnumpy(getattr(result, n))
                for n in ("swdown", "glw", "olr", "rthratensw",
                          "rthratenlw")}

    unmasked = run(specified=False, nc=written["nc"], updates=3,
                   effc=written["effc"], effi=written["effi"],
                   effs=written["effs"])
    fixed = run(specified=True, nc=written["nc"], updates=3,
                effc=written["effc"], effi=written["effi"],
                effs=written["effs"])
    # The reference for the ring: the same moments, radii reconstructed
    # from them with Morrison's droplet number (the adapter's own path).
    reference = run(specified=False, nc=250e6 / rho, updates=0,
                    effc=zeros, effi=zeros, effs=zeros)
    for name in fixed:
        tile = (Ellipsis, slice(1, -1), slice(1, -1))
        np.testing.assert_array_equal(
            fixed[name][tile].view(np.uint32),
            unmasked[name][tile].view(np.uint32), err_msg=name)
        ring_values = fixed[name][..., ring]
        np.testing.assert_allclose(ring_values, reference[name][..., ring],
                                   rtol=2e-5, atol=1e-6, err_msg=name)
    assert not np.allclose(unmasked["swdown"][ring],
                           reference["swdown"][ring], rtol=1e-3)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("workspace", [False, True])
def test_olr_is_the_column_top_upward_longwave(monkeypatch, workspace):
    cp = pytest.importorskip("cupy")
    from gpuwm.core import rrtmgp as rr
    from gpuwm.core.model import SharedRRTMGPChunkWorkspace

    atmosphere, fields, play, *_rest = _column_case(cp)
    ptop = _rest[3]
    nz, ny, nx = play.shape
    chunk = 16
    tops, model_tops = [], []
    original = rr._store_model_flux_pair

    def record(up, dn, model_nlay, out_up, out_dn, **kw):
        if kw.get("daylight") is None and len(tops) < -(-ny * nx // chunk):
            tops.append(cp.asnumpy(up[:, -1]))
            model_tops.append(cp.asnumpy(up[:, model_nlay]))
        return original(up, dn, model_nlay, out_up, out_dn, **kw)

    monkeypatch.setattr(rr, "_store_model_flux_pair", record)
    radiation = rr.RRTMGPRadiation(
        datetime(2023, 6, 21, 6), cp.full((ny, nx), 35.0),
        cp.full((ny, nx), -90.0), column_chunk=chunk, shortwave=False)
    if workspace:
        radiation.chunk_workspace = SharedRRTMGPChunkWorkspace(
            nz=nz, column_chunk=chunk, p_top=ptop)
    state = SimpleNamespace(elapsed_seconds=0.0, p_top=ptop,
                            qc=atmosphere["qc"],
                            qr=cp.zeros_like(atmosphere["qc"]))
    cfg = SimpleNamespace(mp_physics=1, dt=60.0, radt=12.0, radt_minutes=12.0)
    result = radiation(atmosphere=atmosphere, fields=fields, state=state,
                       cfg=cfg)
    column_top = np.concatenate(tops).reshape(ny, nx)
    model_top = np.concatenate(model_tops).reshape(ny, nx)
    olr = cp.asnumpy(result.olr)
    np.testing.assert_array_equal(olr, column_top)
    assert np.abs(column_top - model_top).max() > 0.5
