"""Full-field identity of the default clock through the real GPU executor."""
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def test_unchanged_default_clock_preserves_every_full_field_hash():
    import cupy as cp
    from gpuwm import terrain_clock as tc
    from gpuwm.core.model import execute_experiment
    from tilestream import harness
    from test_model_stability_gpu import _model
    from test_terrain_clock import _acoustic

    hashes, clocks = [], []
    for mode in ("measured", "local_face"):
        exp, model = _model(False)
        domain = exp.root
        run = domain.run if mode == "local_face" else replace(
            domain.run, terrain_clock=mode)
        exp = replace(exp, domains=(replace(domain, run=run),))
        state = model.root.state
        start = tc.StartWinds.from_fields("d01", *[
            cp.asnumpy(getattr(state, name)) for name in ("u", "v", "php", "phb")])
        adapted, decisions = tc.clock_for_domains(exp, _acoustic(exp, {1: 0.}),
            statics={1: {"HGT_M": np.zeros((run.ny, run.nx))}},
            starts={1: start}, announce=False)
        assert adapted is exp
        assert decisions[0].dt == 1 and decisions[0].time_step_sound == run.time_step_sound
        if mode == "local_face":
            assert run.terrain_clock == "local_face"
            assert decisions[0].faces_read is not None
            assert decisions[0].local_note is None
        model.root.cfg = adapted.root
        report = execute_experiment(model, experiment=adapted,
                                    pool_trim_per_period=False)
        assert report.steps == 4
        cp.cuda.Stream.null.synchronize()
        hashes.append(harness.hash_field_map(state))
        clocks.append(tc.clock_receipt(decisions))
    assert hashes[0] and hashes[0] == hashes[1]
    receipt = os.environ.get("TERRAIN_CLOCK_HASH_RECEIPT")
    if receipt:
        Path(receipt).write_text(json.dumps({
            "strict": os.environ.get("GPUWM_WRF_EXACT") == "1",
            "steps": 4, "fields": len(hashes[0]), "hashes": hashes,
            "clock": clocks, "identical": True}, indent=2) + "\n")
