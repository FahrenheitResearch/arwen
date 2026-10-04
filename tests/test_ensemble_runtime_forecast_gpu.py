"""Actual config-driven ordinary forecasts against shared member preparation.

The source decoder, real initialization, physics, clocks and tile steppers are
the original runtime processors. Read-only binding instrumentation observes
their actual owners. A real supplied case-data config is required.
"""
from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
import gc
import json
import os
from pathlib import Path

import pytest

from conftest import requires_gpu
from test_ensemble_production_forecast_gpu import (_array_word_record, _clock_words,
    _collector, _driver_arrays, _history_words, _json_value, _sha)

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.fixture(scope="module")
def runtime_case():
    source = os.environ.get("GPUWM_TEST_RUNTIME_CASE_CONFIG")
    if not source:
        pytest.skip("set GPUWM_TEST_RUNTIME_CASE_CONFIG to an actual staged case-data config")
    from gpuwm.case_data import load_experiment_case
    exp, data = load_experiment_case(Path(source))
    assert len(exp.domains) == 1, "the runtime gate derives its own explicit nested control"
    assert len(data.forcing) >= 1
    assert all(Path(path).is_file() for path in data.forcing)
    assert Path(data.vtable).is_file() and Path(data.wps_namelist).is_file()
    assert Path(data.geog_root).is_dir()
    return exp, data


def _variant(base, variant):
    from gpuwm.core.streaming import StreamingOptions
    from gpuwm.experiment import DomainConfig
    root = base.root
    cfg = replace(root.run, dt=12., clock_dt=0., run_seconds=48.,
                  use_adaptive_time_step=variant == "adaptive")
    if variant == "tile":
        from gpuwm.core.streaming import _halo_for
        halo = _halo_for(cfg)
        cfg = replace(cfg, nx=max(cfg.nx, 80, 4 * halo), ny=max(cfg.ny, 72, 4 * halo))
    root = replace(root, run=cfg, time_step=12, time_step_fract_num=0,
                   time_step_fract_den=1, history_interval_s=12., history_begin_s=12.,
                   history_end_s=None, tiles=None)
    domains = [root]
    if variant == "nested":
        child = replace(cfg, grid_id=2, nx=24, ny=24, dx=cfg.dx / 3., dy=cfg.dy / 3.,
                        dt=4., nested=True, specified=False)
        domains.append(DomainConfig(grid_id=2, parent_id=root.grid_id,
            i_parent_start=5, j_parent_start=5, parent_grid_ratio=3,
            parent_time_step_ratio=3, history_interval_s=12., history_begin_s=12.,
            run=child, start_time=base.start_time))
    tiles = (StreamingOptions(mode="on", store="host", tile_nx=cfg.nx // 2,
                             tile_ny=cfg.ny // 2, nbuffers=1)
             if variant == "tile" else StreamingOptions(mode="off"))
    return replace(base, domains=tuple(domains), run_seconds=48., restart_interval_s=0., tiles=tiles)


def _case_data_variant(base, experiment, data, variant, output):
    if variant not in ("nested", "tile"):
        return data
    from gpuwm.companion_domains import candidate_wps_text
    config = Path(output) / (variant + ".toml")
    path = config.with_suffix(".namelist.wps")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {"case_data": {"forcing_interval_s": data.forcing_interval_s}}
    if data.forcing_interval_s is None:
        raw["case_data"].pop("forcing_interval_s")
    text = candidate_wps_text(raw, base, experiment, config, original_wps=data.wps_namelist)
    path.write_text(text, encoding="utf-8", newline="\n")
    # Both original and shared preparations consume this actual derived WPS.
    # The unchanged source/Vtable authorities remain attached to the case.
    return replace(data, wps_namelist=path, authority_identity_wps_namelist=path)


class _RuntimeWords:
    def __init__(self, root, *, reference=None):
        self.root, self.reference = Path(root), reference
        self.root.mkdir(parents=True, exist_ok=True)
        self.records, self.owners = {}, {}
        self.pure_member = 0

    def _member(self):
        from gpuwm.ensemble.runtime_context import current_capture
        capture = current_capture()
        return self.pure_member if capture is None else capture.member_id

    def bind_state(self, *, prepared_case, state, cfg, grid, clock=None):
        member = self._member()
        self.owners[(member, 1)] = (state, cfg, clock)
        self.snapshot("initialized", member=member, grid_id=1)

    def bind_model(self, model):
        member = self._member()
        for node in model.walk_parent_first():
            self.owners[(member, int(node.cfg.grid_id))] = (node.state, node.cfg.run, node.clock)
            self.snapshot("initialized", member=member, grid_id=int(node.cfg.grid_id))

    def _arrays(self, key, group, values):
        row = self.records.setdefault(key, {})
        assert group not in row, (key, group, "duplicate capture")
        words = {name: _array_word_record(value) for name, value in sorted(values.items())}
        if self.reference is not None:
            expected = self.reference.records[key][group]
            assert set(words) == set(expected), (key, group, "inventory", set(words) ^ set(expected))
            for name in words:
                assert words[name] == expected[name], (key, group, name, words[name], expected[name])
        row[group] = words

    def snapshot(self, phase, *, member, grid_id):
        from gpuwm.core.device_inventory import state_array_shapes
        state, cfg, clock = self.owners[(member, grid_id)]
        key = f"member-{member:04d}/d{grid_id:02d}/{phase}"
        stream = getattr(state, "_streamed_domain", None)
        if stream is None:
            self._arrays(key, "state", {name: getattr(state, name) for name in state_array_shapes(cfg)})
            self._arrays(key, "physics", _driver_arrays(state.physics))
            controls = {"call_counts": dict(state.physics.call_counts),
                        "microphysics_updates": int(state.physics.microphysics_updates)}
        else:
            # Read the actual typed carrier store. The resident preparation
            # state is stale; rebuilt per-tile scratch is not a domain field.
            run = getattr(stream, "_run", None)
            drain = getattr(run, "drain", None)
            if callable(drain):
                drain()
            self._arrays(key, "carried_store", stream.store)
            controls = {"carried_clock": _json_value(dict(stream.scalars))}
        controls["clock"] = None if clock is None else _clock_words(clock)
        if self.reference is not None:
            assert controls == self.reference.records[key]["controls"], (key, "controls")
        self.records[key]["controls"] = controls

    def history(self, **frame):
        from gpuwm.io.wrfout import _device_state_frame
        member, grid_id = frame["member_id"], frame["grid_id"]
        phase = "history-" + frame["valid_time"].strftime("%Y-%m-%d_%H:%M:%S")
        self.snapshot(phase, member=member, grid_id=grid_id)
        key = f"member-{member:04d}/d{grid_id:02d}/{phase}"
        stream = frame["streamed"]
        if stream is None:
            output = _device_state_frame(frame["state"], include_diagnostic_pressure=True)
            if frame["refl_field"] is not None:
                output["REFL_10CM"] = frame["refl_field"]
        else:
            output = stream.history_fields()
            materialize = getattr(output, "materialize", None)
            if callable(materialize):
                output = materialize()
        assert isinstance(output, Mapping)
        self._arrays(key, "output", output)
        self._arrays(key, "metadata", frame["metadata"])

    def progress(self, **event):
        elapsed = event.get("member_model_elapsed_seconds", event.get("model_elapsed_seconds"))
        if event.get("phase") != "post-d01-sync" or elapsed != 48.:
            return
        member = int(event.get("member_id", self._member()))
        for owner_member, grid_id in tuple(self.owners):
            if owner_member == member:
                self.snapshot("final", member=member, grid_id=grid_id)
                del self.owners[(member, grid_id)]

    def flush(self):
        assert not self.owners, "every initialized owner must reach its actual final committed step"
        path = self.root / "runtime-word-receipt.json"
        path.write_text(json.dumps(_json_value(self.records), indent=2, sort_keys=True) + "\n")
        return {"path": str(path), "sha256": _sha(path), "snapshots": len(self.records)}


class _RuntimeProducts:
    def __init__(self, collector, words):
        self.collector, self.words = collector, words

    def __getattr__(self, name):
        return getattr(self.collector, name)

    def submit(self, **frame):
        self.words.history(**frame)
        return self.collector.submit(**frame)


def _products(collector, grids):
    collector.finish_run()
    receipt = collector.require_complete()
    assert not receipt["pending_rosters"] and not receipt["unavailable_products"]
    assert len(collector.spools) == grids
    result = {}
    for spool in collector.spools.values():
        assert len(spool.frames) == 4
        for valid, frame in sorted(spool.frames.items()):
            assert frame["status"] == "complete" and not frame["unavailable_fields"]
            assert len(frame["products"]) == 1 and len(frame["maps"]) == 3
            for kind in ("products", "maps"):
                for relative in frame[kind]:
                    path = spool.root / relative
                    result[f"{spool.domain}/{valid}/{kind}/{path.name}"] = _sha(path)
    return result


@pytest.mark.parametrize("variant,members", [("fixed", 1), ("fixed", 4),
    ("adaptive", 4), ("nested", 4), ("tile", 4)])
def test_actual_runtime_members_match_original_source_processor(runtime_case, variant, members, tmp_path, monkeypatch):
    from gpuwm import runtime
    from gpuwm.ensemble import runtime_context
    from gpuwm.ensemble.production import PreparedEnsembleSession
    from gpuwm.ensemble.runtime_preparation import RuntimeMemberInputs
    from gpuwm.ensemble.runtime_context import MemberOutputCapture, ensemble_scope, member_output_scope
    from test_ensemble_batch_product_spool_gpu import _renderer
    exp = _variant(runtime_case[0], variant)
    data = _case_data_variant(runtime_case[0], exp, runtime_case[1], variant, tmp_path / "case")
    renderer = _renderer()
    reference = _RuntimeWords(tmp_path / "ordinary-words")
    observed = [reference]
    original_single = runtime_context.bind_current_member_state
    original_tree = runtime_context.bind_current_member_model
    def single(**owner):
        original_single(**owner)
        observed[0].bind_state(**owner)
    def tree(model):
        original_tree(model)
        observed[0].bind_model(model)
    # Instrument only the explicit owner-binding notification. Every source,
    # initializer, clock, physics and forecast function remains original.
    monkeypatch.setattr(runtime_context, "bind_current_member_state", single)
    monkeypatch.setattr(runtime_context, "bind_current_member_model", tree)
    inputs = RuntimeMemberInputs(exp, data)
    control_collector = _collector(tmp_path / "ordinary-products", inputs, members, renderer, keep=members == 1)
    control = _RuntimeProducts(control_collector, reference)
    summaries = []
    for member in range(members):
        reference.pure_member = member
        with member_output_scope(MemberOutputCapture(control.submit, member, members == 1)):
            summaries.append(runtime.run_experiment(exp, data, tmp_path / "ordinary" / f"member-{member:04d}",
                                                   progress_callback=reference.progress))
        gc.collect()
    expected_products = _products(control_collector, len(exp.domains))
    reference_receipt = reference.flush()
    if members == 1:
        # A separate pure ordinary route proves that capture and optional
        # member retention preserve the entire original history artifact.
        plain = _RuntimeWords(tmp_path / "plain-words")
        observed[0] = plain
        runtime.run_experiment(exp, data, tmp_path / "plain", progress_callback=plain.progress)
        history = _history_words(tmp_path / "plain")
        assert _history_words(tmp_path / "ordinary") == history
        expected_plain = {key: row for key, row in reference.records.items()
                          if key.endswith(("/initialized", "/final"))}
        assert plain.records == expected_plain
        plain.flush()
        gc.collect()
    words = _RuntimeWords(tmp_path / "shared-words", reference=reference)
    observed[0] = words
    collector = _collector(tmp_path / "shared", inputs, members, renderer, keep=members == 1)
    session = PreparedEnsembleSession({"members": members, "keep_member_files": members == 1},
        output_directory=collector.root, collector=_RuntimeProducts(collector, words),
        identical_members="identity gate: every member must reproduce the ordinary forecast word for word")
    with ensemble_scope(session):
        result = runtime.run_experiment(exp, data, collector.root, progress_callback=words.progress)
    assert result.completed_seconds == 48. and result.nan_free
    manifest = json.loads((collector.root / "ensemble-run.json").read_text())
    assert manifest["status"] == "PASS"
    assert manifest["members_completed"] == list(range(members))
    counts = manifest["shared_native_preparation"]["counts"]
    assert counts["catalog_builds"] == counts["forcing_decodes"] == counts["root_preparations"] == 1
    assert counts["root_restores"] == members - 1
    if variant == "nested":
        assert counts["child_input_preparations"] == 1
        assert counts["child_input_reuses"] == members - 1
    assert not words.owners
    assert set(words.records) == set(reference.records)
    assert _products(collector, len(exp.domains)) == expected_products
    if members == 1:
        assert _history_words(collector.root / "members") == history
    else:
        assert not manifest["member_history_files"]
        assert not list(collector.root.rglob("wrfout_*"))
    record = {"schema": "gpuwm-ensemble-runtime-source-identity-gate.v1", "status": "PASS",
        "variant": variant, "members": members, "ordinary_words": reference_receipt,
        "shared_words": words.flush(), "shared_preparation": manifest["shared_native_preparation"],
        "products": expected_products, "counter_calendars": [row.get("result", {}).get("ensemble_counter_observations")
            for row in manifest["member_results"]]}
    (tmp_path / "runtime-identity-receipt.json").write_text(json.dumps(_json_value(record), indent=2) + "\n")
