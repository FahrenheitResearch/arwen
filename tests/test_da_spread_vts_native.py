"""Native byte-identity retention probe; no weather or skill judgment."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu


@requires_gpu
def test_native_shifted_slots_restore_every_central_restart_array(tmp_path, monkeypatch):
    from gpuwm.da import spread_vts
    from gpuwm.da.spread_vts_ram import get
    from tools import da_member_leg, da_member_leg_proof as proof
    from tools.da_cycle_prepared import write_leg_restart
    from gpuwm.io.restart import tree_restart_members, read_restart_header
    from datetime import timedelta

    context, _ = proof.fixture_context(dict(leg=0, absolute_leg_number=0,
        t_start=0, t_end=1800, nested=False), output=tmp_path/"member")
    from test_da_cycle_join_gpu import _experiment_to, _wire_parent
    exp = _experiment_to(2700)
    root = replace(exp.root, history_interval_s=300,
                   run=replace(exp.root.run, output_interval_s=300))
    exp = replace(exp, domains=(root,), acknowledgements=context.inputs.experiment.acknowledgements,
                  constant_glw_wm2=context.inputs.experiment.constant_glw_wm2)
    context.inputs.experiment = exp
    context.inputs.boundary_interval_seconds = 10800
    context.args.history_interval_seconds = 300
    context.args.spread_repair_vtsm = True
    context.args.spread_vts_grid_identity = "a"*64
    cycle = spread_vts.VtsCycle(tmp_path/"s", max_bytes=256*1024**2)
    context.args.spread_vts_socket = str(cycle.server.path)
    comparisons = []
    real_continue = spread_vts.continue_and_restore

    def measured_continue(model, **arguments):
        result = real_continue(model, **arguments)
        after = write_leg_restart(model, tmp_path/"restored",
            valid_time=exp.start_time+timedelta(seconds=1800), auto_epssm=exp.auto_epssm)
        originals = tree_restart_members(arguments["restart"])
        restored = tree_restart_members(after)
        assert set(originals) == set(restored)
        for gid in originals:
            original_header = read_restart_header(originals[gid])
            restored_header = read_restart_header(restored[gid])
            for clock_name in proof.CLOCK_FIELDS:
                assert original_header[clock_name] == restored_header[clock_name], clock_name
            # Metadata text can record a new publication time. Every numeric
            # state, physics driver, soil, accumulator and clock array is exact.
            with np.load(originals[gid], allow_pickle=False) as before, np.load(restored[gid], allow_pickle=False) as after_arrays:
                names = {name for name in before.files if name != "__gpuwm_restart_header__" and before[name].dtype.kind in "biuf"}
                assert names == {name for name in after_arrays.files if name != "__gpuwm_restart_header__" and after_arrays[name].dtype.kind in "biuf"}
                assert names
                for name in names:
                    assert before[name].dtype == after_arrays[name].dtype
                    assert before[name].shape == after_arrays[name].shape
                    assert before[name].tobytes() == after_arrays[name].tobytes(), name
                comparisons.append({"domain": int(gid), "arrays": len(names)})
        return result

    monkeypatch.setattr(spread_vts, "continue_and_restore", measured_continue)
    def wire(stop_seconds, *, child_dc=None):
        assert child_dc is None
        wired = _wire_parent(replace(exp, run_seconds=stop_seconds))
        wired["root"].state.qr[:, 8:25, 8:25] = 2e-4
        return SimpleNamespace(node=wired["root"], restored=SimpleNamespace(
            surface=wired["surface"], initial_result=SimpleNamespace(state=wired["root"].state)),
            driver=wired["driver"], clocks=wired["clocks"], schedule=wired["schedule"], child_dc=None)
    try:
        result = da_member_leg.run_member_leg(context, 0, wire_override=wire)
        references = result.record["spread_vts"]["slots"]
        assert [slot["metadata"]["valid_ticks"]/slot["metadata"]["tick_den"] for slot in references] == [900, 1800, 2700]
        fingerprints = []
        for reference in references:
            with get(cycle.server.path, reference["key"]) as (fields, record):
                from hashlib import sha256
                fingerprints.append(sha256(fields["state/thp"].tobytes()+fields["state/u"].tobytes()).hexdigest())
                assert record["metadata"]["forecast_origin_seconds"] == 0
                assert record["metadata"]["observation_cutoff_seconds"] == 0
                assert set(name for name in fields if name.startswith("surface/")) == {
                    "surface/t2", "surface/u10", "surface/v10", "surface/q2", "surface/psfc"}
        assert len(set(fingerprints)) == 3, "actual forecasts must differ at all three source clocks"
        assert comparisons and result.record["elapsed_seconds"] == 1800
        proof.write_json(tmp_path/"native-retention-receipt.json", {
            "scope": "native full-state numeric byte identity; no weather qualification",
            "clocks_seconds": [900, 1800, 2700], "sample_sha256": fingerprints,
            "central_arrays_compared": comparisons, "producer": result.record["spread_vts"]})
    finally:
        cycle.clear()
