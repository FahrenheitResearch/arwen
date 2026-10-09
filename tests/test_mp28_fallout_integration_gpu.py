"""Twenty real dycore steps with seeded fallout, whole-field hash identity."""
import hashlib
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu


def test_twenty_dycore_steps_keep_all_field_hashes(monkeypatch):
    cp = pytest.importorskip("cupy")
    from gpuwm.core import dycore, thompson_aerosol_sed as sed
    from gpuwm.core.physics import initialize_physics
    import test_mp28_forecast_smoke as g4
    cfg = g4._forecast_config()
    fields = g4._FINITE_FIELDS + ("p", "nr", "ni")
    def run(parallel):
        monkeypatch.setattr(sed, "LEVEL_PARALLEL_FALLOUT", parallel)
        state, forcing = g4._build_states(cp, cfg, bubble=True, wind=0.)
        g4._attach_specified_boundaries(state, forcing, cfg)
        initialize_physics(state, cfg)
        state.qs[8:20, 5:-5, 5:-5] = cp.float32(1e-4)
        state.qg[8:15, 5:-5, 5:-5] = cp.float32(1e-4)
        state.qr[3:12, 5:-5, 5:-5] = cp.float32(1e-5)
        state.nr[3:12, 5:-5, 5:-5] = cp.float32(1e4)
        frames = []
        for step in range(20):
            dycore.step(state, cfg)
            cp.cuda.Stream.null.synchronize()
            frame = {}
            for name in fields:
                value = getattr(state, name, None)
                if value is not None:
                    assert bool(cp.isfinite(value).all()), (step, name)
                    frame[name] = hashlib.sha256(cp.asnumpy(value).tobytes()).hexdigest()
            frames.append(frame)
        assert float(state.qg.max()) > 0.
        assert float(state.qs.max()) > 0.
        return frames
    reference = run(False)
    actual = run(True)
    receipt = {"steps": 20, "shape": [cfg.nz, cfg.ny, cfg.nx], "dt": cfg.dt,
               "strict": os.environ.get("GPUWM_WRF_EXACT", "") == "1",
               "serial": reference, "parallel": actual,
               "identical": reference == actual}
    if os.environ.get("GPUWM_MP28_HASH_RECEIPT"):
        Path(os.environ["GPUWM_MP28_HASH_RECEIPT"]).write_text(json.dumps(receipt, indent=2)+"\n")
    assert actual == reference
