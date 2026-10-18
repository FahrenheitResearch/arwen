"""Grade the production KF adapter handoff on the committed WRF columns.

This checks the adapter and its W0AVG history, not whole-model coupling.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from conftest import requires_gpu


@requires_gpu
def test_production_adapter_keeps_the_oracle_words_and_updates_history_once():
    import cupy as cp
    from gpuwm.core.kf import KainFritsch
    from tools.kf_wrf461_oracle import kf_columns

    fixture = Path(__file__).parent / "data/oracles/kf"
    columns = kf_columns.read_columns(fixture / "columns-structured.bin")
    outputs = kf_columns.read_outputs(fixture / "oracle-structured-O0.out")
    words = graded = 0
    for column, expected in zip(columns, outputs):
        nz, warm, fqr, fqi, fqs, _stepcu, adaptive, _trigger = column["ints"]
        # No admitted microphysics option has QS without QI. That column
        # branch is covered by the existing direct-launcher oracle.
        if fqs and not fqi:
            continue
        mp = 1 if warm else 6 if fqi and fqs else 50 if fqr else 0
        dt, dx, cudt = map(float, column["scal"][:3])

        def device(name):
            return cp.asarray(column[name][:, None, None], dtype=cp.float32)

        atmosphere = {name: device(source) for name, source in (
            ("u", "u"), ("v", "v"), ("temperature", "t"), ("qv", "qv"),
            ("pressure", "p"), ("rho", "rho"), ("exner", "pi"), ("dz", "dz"))}
        state = SimpleNamespace(w=device("w"), elapsed_seconds=0.0)
        cfg = SimpleNamespace(mp_physics=mp, dt=dt / 2, clock_dt=dt,
                              dx=dx, cudt_minutes=cudt,
                              use_adaptive_time_step=bool(adaptive))
        adapter = KainFritsch()
        adapter.ensure_trigger_history(state)
        adapter.w0avg[...] = device("w0avg")
        adapter.update_trigger_history(state=state, cfg=cfg)
        advanced = cp.asnumpy(adapter.w0avg).copy()
        result = adapter(atmosphere=atmosphere, fields={}, state=state, cfg=cfg)
        assert np.array_equal(cp.asnumpy(adapter.w0avg).view(np.uint32), advanced.view(np.uint32))
        for field in ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten", "rqscuten"):
            value = getattr(result, field)
            if value is None:
                assert (field == "rqicuten" and not fqi) or (field == "rqscuten" and not fqs)
                continue
            got = cp.asnumpy(value)[:, 0, 0]
            ref = expected[field]
            np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32), err_msg=field)
            words += got.size
        np.testing.assert_array_equal(advanced[:, 0, 0].view(np.uint32),
                                      expected["w0avg"].view(np.uint32))
        for name, source in (("pratec", "pratec"), ("rainc", "raincv")):
            got = cp.asnumpy(getattr(result, name))[0, 0]
            assert got.tobytes() == np.float32(expected[source]).tobytes(), name
            words += 1
        nca = np.float32(expected["nca"])
        if nca.tobytes() == np.float32(column["scal"][3]).tobytes():
            nca = np.float32(0)
        assert cp.asnumpy(result.nca_seconds)[0, 0].tobytes() == nca.tobytes()
        words += 1
        graded += 1
    assert graded == 123
    assert words > 25000
