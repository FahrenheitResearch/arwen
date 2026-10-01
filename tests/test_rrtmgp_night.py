"""Bitwise SW driver checks for daylight compaction."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("validation", ["fused", "full"])
@pytest.mark.parametrize("workspace", [False, True])
@pytest.mark.parametrize("uniform_top", [False, True])
@pytest.mark.parametrize("light", ["dark", "mixed", "day"])
def test_daylight_compaction_flux_bits(monkeypatch, workspace, uniform_top, light, validation):
    cp = pytest.importorskip("cupy")
    from gpuwm.core import rrtmgp as rr, rrtm_lw
    from gpuwm.core.model import SharedRRTMGPChunkWorkspace

    is_capturing = rrtm_lw._stream_is_capturing
    nz, ny, nx, chunk = 12, 1, 13, 4
    plev = np.broadcast_to(
        np.geomspace(100000.0, 5718.0, nz + 1)[:, None, None],
        (nz + 1, ny, nx)).copy()
    if not uniform_top:
        plev[-1, 0] += (np.arange(nx) % 3) * 0.00048828125
    play = np.sqrt(plev[:-1] * plev[1:])
    temperature = np.broadcast_to(
        np.linspace(290.0, 215.0, nz)[:, None, None], play.shape).copy()
    temperature += np.arange(nx)[None, None, :] * 0.125
    exner = (play / 100000.0) ** (287.0 / 1004.0)
    qv = np.broadcast_to(np.geomspace(8e-3, 1e-5, nz)[:, None, None],
                         play.shape).copy()
    qv *= 1.0 + np.arange(nx)[None, None, :] * 0.01
    qc = np.zeros_like(play)
    qi = np.zeros_like(play)
    qc[3:7] = 2e-4
    qi[7:10] = 1e-4
    atmosphere = {name: cp.asarray(value, dtype=cp.float32)
                  for name, value in {
                      "pressure": play, "p_interface": plev,
                      "temperature": temperature, "theta": temperature / exner,
                      "exner": exner, "qv": qv, "qc": qc, "qi": qi}.items()}
    fields = {"tsk": cp.full((ny, nx), 288.0, cp.float32),
              "albedo": cp.asarray(np.linspace(0.1, 0.3, nx)[None],
                                    dtype=cp.float32),
              "emiss": cp.full((ny, nx), 0.96, cp.float32)}
    state = SimpleNamespace(elapsed_seconds=0.0, qc=atmosphere["qc"],
                            qr=cp.zeros_like(atmosphere["qc"]))
    cfg = SimpleNamespace(mp_physics=1, dt=60.0, radt=12.0, radt_minutes=12.0)
    radiation = rr.RRTMGPRadiation(
        datetime(2011, 4, 27, 18), cp.full((ny, nx), 35.0),
        cp.full((ny, nx), -97.5), column_chunk=chunk,
        validation_mode=validation)
    if workspace:
        radiation.chunk_workspace = SharedRRTMGPChunkWorkspace(
            nz=nz, column_chunk=chunk, p_top=5718.0)
    mu = {"dark": [-0.5] * nx, "day": [0.5] * nx,
          "mixed": [0.7, -0.2, 0.0, 0.2, -0.5, 0.6, 0.1,
                    -0.1, 0.8, -0.3, 0.4, -0.9, 0.3]}[light]
    radiation._cosine_zenith = lambda *a, **kw: cp.asarray([mu], cp.float32)
    snapshots = []
    prepare_widths = []
    original_fluxes = rr._fluxes_to_radiation
    original_prepare = rr._prepare_above_model_chunk

    def record(lw_up, lw_dn, sw_up, sw_dn, *a, **kw):
        snapshots.append((cp.asnumpy(sw_up).view(np.uint32),
                          cp.asnumpy(sw_dn).view(np.uint32)))
        return original_fluxes(lw_up, lw_dn, sw_up, sw_dn, *a, **kw)

    def prepare(**kw):
        result = original_prepare(**kw)
        if kw["kind"] == "sw":
            prepare_widths.append(result.profile.play.shape[0])
        return result

    monkeypatch.setattr(rr, "_fluxes_to_radiation", record)
    monkeypatch.setattr(rr, "_prepare_above_model_chunk", prepare)
    # Emulate capture only at the predicate. This selects the unchanged
    # full-column loop and the general above-model rows as the reference.
    monkeypatch.setattr(rrtm_lw, "_stream_is_capturing", lambda xp: True)
    radiation(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    assert prepare_widths == [4, 4, 4, 1]
    prepare_widths.clear()
    scratch_keys = set(rr._CHUNK_SCRATCH)
    monkeypatch.setattr(rrtm_lw, "_stream_is_capturing", lambda xp: False)
    radiation(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    if workspace and light == "mixed":
        assert set(rr._CHUNK_SCRATCH) == scratch_keys
    for reference, compacted in zip(snapshots[0], snapshots[1]):
        np.testing.assert_array_equal(reference, compacted)
        assert np.all(compacted[np.asarray(mu) <= 0] == 0)
    assert prepare_widths == {"dark": [], "mixed": [4, 3],
                              "day": [4, 4, 4, 1]}[light]
    prepare_widths.clear()
    radiation(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    for reference, repeated in zip(snapshots[1], snapshots[2]):
        np.testing.assert_array_equal(reference, repeated)

    if workspace and uniform_top and validation == "fused":
        from gpuwm.core.health_ledger import HealthLedger, deferring

        state.p_top = 5718.0
        mu_device = cp.asarray([mu], cp.float32)
        radiation._cosine_zenith = lambda *a, **kw: mu_device
        captured_fluxes = []

        def device_record(lw_up, lw_dn, sw_up, sw_dn, *a, **kw):
            captured_fluxes.append((sw_up.copy(), sw_dn.copy()))
            return original_fluxes(lw_up, lw_dn, sw_up, sw_dn, *a, **kw)

        monkeypatch.setattr(rr, "_fluxes_to_radiation", device_record)
        monkeypatch.setattr(rrtm_lw, "_stream_is_capturing", is_capturing)
        stream = cp.cuda.Stream(non_blocking=True)
        cp.cuda.get_current_stream().synchronize()
        ledger = HealthLedger()
        with stream:
            stream.begin_capture()
            with deferring(ledger):
                radiation(atmosphere=atmosphere, fields=fields, state=state,
                          cfg=cfg)
            graph = stream.end_capture()
            graph.launch(stream)
        stream.synchronize()
        ledger.drain()
        for reference, captured in zip(snapshots[0], captured_fluxes[0]):
            np.testing.assert_array_equal(reference,
                                          cp.asnumpy(captured).view(np.uint32))
