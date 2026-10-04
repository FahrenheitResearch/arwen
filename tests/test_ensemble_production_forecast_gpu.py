"""Real prepared-tree clocks, history words and native ensemble products.

This gate advances the ordinary source processor, not a recreated scalar
dycore. N=1 also compares complete original history files. N=4 must enter the
initialized native executor; the separate fallback arm advances the ordinary
runner once for each member through the production session.
"""
from dataclasses import fields, is_dataclass, replace
from contextlib import nullcontext
from collections.abc import Mapping
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pytest

from conftest import requires_gpu
from test_ensemble_batch_physics_init_gpu import card_free_of_earlier_tests  # noqa: F401

pytestmark = [pytest.mark.gpu, requires_gpu]

_SEED = 2026100200
_CLOCK_FIELDS = ("ticks", "step_ticks", "tick_den", "run_ticks", "step_count",
                 "dt_fp32", "dtbc_fp32")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(name): _json_value(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


@pytest.fixture(scope="module")
def production_real(tmp_path_factory):
    directory = os.environ.get("GPUWM_TEST_WRF_REAL_DIRECTORY")
    if not directory:
        pytest.skip("set GPUWM_TEST_WRF_REAL_DIRECTORY to retained real WRF inputs")
    from gpuwm.wrfinput_door import resolve_wrfinput_run
    from gpuwm.wrfinput_forecast import prepare_wrf_run
    from gpuwm.prepared_domain_tree_forecast import _with_terrain_acoustics
    run = resolve_wrfinput_run(Path(directory))
    text, count = re.subn(r"(?m)^(\s*history_interval_s)\s*=.*$",
                         r"\1 = 12.0", run.toml_text)
    assert count, "the imported configuration must contain history authority"
    inputs = prepare_wrf_run(replace(run, toml_text=text),
                             tmp_path_factory.mktemp("production-ensemble-source"), run_seconds=24)
    inputs = _with_terrain_acoustics(inputs)
    cfg = inputs.experiment.root.run
    assert len(inputs.experiment.domains) == 1
    assert cfg.dt == 12 and not cfg.use_adaptive_time_step
    assert inputs.experiment.root.history_interval_s == 12
    assert (cfg.mp_physics, cfg.bl_pbl_physics, cfg.sf_sfclay_physics,
            cfg.sf_surface_physics) == (8, 1, 1, 2)
    return inputs


def _renderer():
    from gpuwm import rustwx
    renderer = (Path(os.environ["GPUWM_ENSEMBLE_RENDERER"])
                if "GPUWM_ENSEMBLE_RENDERER" in os.environ else rustwx.find_renderer())
    if renderer is None:
        pytest.fail("the production gate requires the built native ensemble renderer")
    from gpuwm.io import nc_writer_bridge
    assert nc_writer_bridge.unavailable_reason() is None
    return renderer


def _driver_arrays(driver):
    """Every persistent stock-driver array, including tendency containers."""
    import cupy as cp
    found = {}
    def walk(value, path):
        if isinstance(value, cp.ndarray):
            found[path] = value
        elif isinstance(value, dict):
            for name, child in value.items():
                walk(child, f"{path}/{name}")
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                walk(getattr(value, field.name), f"{path}/{field.name}")
        elif isinstance(value, (tuple, list)):
            for number, child in enumerate(value):
                walk(child, f"{path}/{number}")
    excluded = {"state", "radiation_callable", "cumulus_callable", "noah_params",
                "noahmp_params", "noahmp_geometry", "ruc_params"}
    for name, value in vars(driver).items():
        if name not in excluded and not name.startswith("_ensemble_"):
            walk(value, name)
    return found


def _clock_words(clock):
    return {name: (np.asarray(getattr(clock, name)).dtype.str,
                   np.asarray(getattr(clock, name)).tobytes().hex())
            for name in _CLOCK_FIELDS}


def _array_word_record(array):
    import cupy as cp
    host = cp.asnumpy(array) if isinstance(array, cp.ndarray) else np.asarray(array)
    return {"dtype": host.dtype.str, "shape": list(host.shape),
            "sha256": hashlib.sha256(host.tobytes(order="C")).hexdigest(),
            "bytes": host.nbytes}


def _member_array(array, expected, member, members):
    shape = tuple(expected["shape"])
    if tuple(array.shape) == shape:
        return array
    if tuple(array.shape) == (members,) + shape:
        return array[member]
    assert len(shape) in (2, 3), (array.shape, shape)
    height = shape[-2]
    packed = shape[:-2] + (members * height, shape[-1])
    assert tuple(array.shape) == packed, (array.shape, packed)
    return (array[:, member * height:(member + 1) * height] if len(shape) == 3
            else array[member * height:(member + 1) * height])


class _WordCapture:
    """Capture exact device words without retaining full forecast volumes."""
    def __init__(self, root, cfg, *, reference=None, members=1):
        self.root, self.cfg = Path(root), cfg
        self.root.mkdir(parents=True, exist_ok=True)
        self.reference, self.members = reference, members
        self.records = {}
        self.nodes = {}

    def _arrays(self, key, group, values, *, member=None):
        records = self.records.setdefault(key, {})
        assert group not in records, (key, group, "duplicate word capture")
        expected = None if self.reference is None else self.reference.records[key][group]
        if expected is not None:
            assert set(values) == set(expected), (key, group, "inventory",
                sorted(set(expected) - set(values)), sorted(set(values) - set(expected)))
        rows = {}
        for name, array in sorted(values.items()):
            if member is not None and expected is not None:
                array = _member_array(array, expected[name], member, self.members)
            row = _array_word_record(array)
            if expected is not None and row != expected[name]:
                import cupy as cp
                safe = hashlib.sha256(f"{key}/{group}/{name}".encode()).hexdigest()[:20]
                failure = self.root / f"failed-{safe}.npz"
                np.savez_compressed(failure, actual=cp.asnumpy(array))
                details = {"phase_member": key, "group": group, "field": name,
                           "expected": expected[name], "actual": row,
                           "actual_words": failure.name}
                failure.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")
                pytest.fail(f"{key} {group}/{name} differs: {details}")
            rows[name] = row
        records[group] = rows

    def state(self, phase, *, member, state, driver, clock, packed_member=None):
        from gpuwm.core.device_inventory import state_array_shapes
        key = f"member-{member:04d}/{phase}"
        self._arrays(key, "state", {name: getattr(state, name)
            for name in state_array_shapes(self.cfg)}, member=packed_member)
        self._arrays(key, "physics", _driver_arrays(driver), member=packed_member)
        controls = {"clock": _clock_words(clock),
                    "call_counts": _json_value(driver.call_counts),
                    "microphysics_updates": int(driver.microphysics_updates)}
        if self.reference is not None:
            assert controls == self.reference.records[key]["controls"], (key, "clock/physics counters")
        self.records[key]["controls"] = controls

    def output(self, *, member, valid_time, state, refl_field, metadata):
        from gpuwm.io.wrfout import _device_state_frame
        phase = f"history-{valid_time.strftime('%Y-%m-%d_%H:%M:%S')}"
        key = f"member-{member:04d}/{phase}"
        frame = _device_state_frame(state, include_diagnostic_pressure=True)
        if refl_field is not None:
            frame["REFL_10CM"] = refl_field
        self._arrays(key, "output", frame)
        self._arrays(key, "metadata", metadata)

    def flush(self):
        path = self.root / "word-receipt.json"
        path.write_text(json.dumps(self.records, indent=2, sort_keys=True) + "\n")
        return {"path": str(path), "sha256": _sha(path), "snapshots": len(self.records),
                "arrays": sum(len(row[group]) for row in self.records.values()
                              for group in ("state", "physics", "output", "metadata") if group in row)}


class _CapturedProducts:
    """Instrument the actual synchronous output consumer, retaining no state."""
    def __init__(self, collector, words, *, native=False):
        self.collector, self.words, self.native = collector, words, native

    def __getattr__(self, name):
        return getattr(self.collector, name)

    def submit(self, **frame):
        member, valid = frame["member_id"], frame["valid_time"]
        if not self.native:
            node = self.words.nodes[member]
            self.words.state(f"history-{valid.strftime('%Y-%m-%d_%H:%M:%S')}",
                member=member, state=frame["state"], driver=node.state.physics, clock=node.clock)
        assert frame["streamed"] is None, "this native eligibility gate uses the actual resident path"
        self.words.output(member=member, valid_time=valid, state=frame["state"],
                          refl_field=frame["refl_field"], metadata=frame["metadata"])
        return self.collector.submit(**frame)


def _collector(root, inputs, members, renderer, *, keep=False):
    from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector
    from gpuwm.ensemble.batch_products import FieldProducts
    requests = (FieldProducts("wind10", "m s-1", (10.0,)),
                FieldProducts("temperature2", "K", (293.15,)),
                FieldProducts("rain_total", "mm", (1.0,)))
    return HeadlineDiagnosticCollector(root, members=members, renderer=renderer,
        start_time=inputs.experiment.start_time, requests=requests, keep_member_files=keep,
        available_bytes=_available_bytes, tile_rows=16, render_products=("prob",))


def _available_bytes():
    import cupy as cp
    free, _ = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    return int(free + pool.total_bytes() - pool.used_bytes())


def _initialization(inputs, member):
    """Original real-input initialization plus the existing seeded IC gate."""
    from gpuwm.wrfinput_forecast import WrfInitialization
    from gpuwm.forecast_initialization import DomainInitialization
    from gpuwm.ensemble.batch_perturbation import initialize_member_winds
    class SeededInitialization(WrfInitialization):
        def restore_domain(self, domain, grid, bundle, **options):
            initialized = super().restore_domain(domain, grid, bundle, **options)
            def initialize():
                result = initialized.initialize_physics()
                initialize_member_winds(state=initialized.state, cfg=domain.run,
                    member_indices=(member,), seeds=(_SEED + member,),
                    phase="after_physics_before_step")
                return result
            return DomainInitialization(initialized.initial_result, initialize)
    return SeededInitialization(inputs)


def _ordinary(inputs, out, member, capture, *, try_native_n1=False):
    Path(out).mkdir(parents=True, exist_ok=True)
    from gpuwm.prepared_domain_tree_forecast import run_prepared_tree
    from gpuwm.ensemble.runtime_context import MemberOutputCapture, current_capture, member_output_scope
    def initialized(*, node, **kwargs):
        capture.words.nodes[member] = node
        if try_native_n1:
            from gpuwm.ensemble.prepared_batch import native_prepared_eligibility
            from gpuwm.ensemble.native_forecast import run_initialized_native_ensemble
            eligibility = native_prepared_eligibility(inputs, node, members=1, keep_member_files=True)
            assert not eligibility.eligible
            assert "N=1 keeps the ordinary runner and output writer" in eligibility.reasons
            assert run_initialized_native_ensemble(inputs, node, members=1,
                collector=capture, available_bytes=_available_bytes) is None
        capture.words.state("initialized", member=member, state=node.state,
                            driver=node.state.physics, clock=node.clock)
        return None
    active = current_capture()
    if active is not None:
        assert active.member_id == member and active.callback == capture.submit
        assert active.keep_member_files == capture.keep_member_files
    scope = (nullcontext() if active is not None else member_output_scope(
        MemberOutputCapture(capture.submit, member, capture.keep_member_files)))
    with scope:
        report = run_prepared_tree(inputs, output_directory=out, io_mode="history",
            initialization=_initialization(inputs, member), ensemble_bootstrap=initialized)
    node = capture.words.nodes.pop(member)
    capture.words.state("final", member=member, state=node.state,
                        driver=node.state.physics, clock=node.clock)
    assert node.clock.step_count == 2 and node.clock.elapsed_seconds == 24
    del node
    return report


def _history_words(root):
    files = [path for path in Path(root).rglob("wrfout_*")
             if path.is_file() and path.suffix != ".json"]
    assert files, "the original runner must produce complete member history files"
    return {path.name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
            for path in sorted(files)}


def _product_words(collector):
    receipt = collector.finish_run()
    assert not receipt["pending_rosters"] and not receipt["unavailable_products"]
    collector.require_complete()
    rows = {}
    for spool in collector.spools.values():
        assert len(spool.frames) == 3
        for valid, frame in sorted(spool.frames.items()):
            assert frame["status"] == "complete"
            assert frame["members_received"] == list(range(collector.members))
            assert not frame["unavailable_fields"]
            assert len(frame["products"]) == 1 and len(frame["maps"]) == 3
            for kind in ("products", "maps"):
                for relative in frame[kind]:
                    path = spool.root / relative
                    rows[f"{valid}/{kind}/{path.name}"] = {"bytes": path.stat().st_size,
                                                           "sha256": _sha(path)}
    assert not list((collector.root / ".ensemble-diagnostics").rglob("*.nc"))
    return rows


_COUNTS = [1, 4] + ([10] if os.environ.get("GPUWM_TEST_ENSEMBLE_N10") == "1" else [])


@pytest.mark.parametrize("members", _COUNTS)
def test_actual_prepared_tree_native_and_ordinary_fallback_identity(production_real, members, tmp_path):
    import cupy as cp
    from gpuwm.prepared_domain_tree_forecast import run_prepared_tree
    from gpuwm.wrfinput_forecast import WrfInitialization
    from gpuwm.ensemble.batch_perturbation import initialize_member_winds
    from gpuwm.ensemble.native_forecast import run_initialized_native_ensemble
    from gpuwm.ensemble.prepared_batch import native_prepared_eligibility
    from gpuwm.ensemble.production import PreparedEnsembleSession
    from gpuwm.ensemble.runtime_context import current_capture
    inputs, renderer = production_real, _renderer()
    cfg = inputs.experiment.root.run
    started = time.perf_counter()
    receipt = {"schema": "gpuwm-ensemble-production-identity-gate.v1", "members": members,
        "source_authority_sha256": dict(inputs.authority_sha256),
        "configuration": {name: getattr(cfg, name) for name in
            ("nx", "ny", "nz", "dx", "dt", "mp_physics", "bl_pbl_physics",
             "sf_sfclay_physics", "sf_surface_physics", "ra_physics", "cu_physics")},
        "timing_policy": "original fixed 12 s clock, 24 s forecast, history every 12 s",
        "member_inputs": "existing illustrative uniform-wind seed gate, same inputs on every arm"}
    control_words = _WordCapture(tmp_path / "ordinary-words", cfg)
    control_collector = _collector(tmp_path / "ordinary-products", inputs, members, renderer,
                                   keep=members == 1)
    control = _CapturedProducts(control_collector, control_words)
    for member in range(members):
        _ordinary(inputs, tmp_path / "ordinary" / f"member-{member:04d}", member, control)
        gc.collect()
    expected_products = _product_words(control_collector)
    receipt["ordinary_words"] = control_words.flush()
    receipt["ordinary_products"] = expected_products
    expected_phases = {"initialized", "final"} | {
        f"history-{(inputs.experiment.start_time + timedelta(seconds=second)).strftime('%Y-%m-%d_%H:%M:%S')}"
        for second in (0, 12, 24)}
    assert set(control_words.records) == {
        f"member-{member:04d}/{phase}" for member in range(members) for phase in expected_phases}

    if members == 1:
        # This arm has no capture scope at all. The original writer's entire
        # artifact must match capture with keep_member_files enabled.
        plain = tmp_path / "plain-ordinary"
        plain.mkdir()
        run_prepared_tree(inputs, output_directory=plain, io_mode="history",
                          initialization=_initialization(inputs, 0))
        expected_history = _history_words(plain)
        assert _history_words(tmp_path / "ordinary") == expected_history
        receipt["ordinary_complete_histories"] = expected_history
        gc.collect()
    else:
        native_words = _WordCapture(tmp_path / "native-words", cfg,
                                    reference=control_words, members=members)
        native_collector = _collector(tmp_path / "native-products", inputs, members, renderer)
        native = _CapturedProducts(native_collector, native_words, native=True)
        def native_bootstrap(*, node, **kwargs):
            eligibility = native_prepared_eligibility(inputs, node, members=members)
            assert eligibility.eligible, eligibility.receipt()
            def validation(*, phase, owners, clock, member_ids, **kw):
                for local_member, member in enumerate(member_ids):
                    label = (f"history-{(inputs.experiment.start_time + timedelta(seconds=clock.elapsed_seconds)).strftime('%Y-%m-%d_%H:%M:%S')}"
                             if phase == "history" else phase)
                    native_words.state(label, member=member, state=owners.batch,
                        driver=owners.physics.driver, clock=clock, packed_member=local_member)
                return {"members_checked": list(member_ids), "word_inventory": "all state and persistent physics arrays"}
            report = run_initialized_native_ensemble(inputs, node, members=members,
                member_ids=tuple(range(members)), member_seeds=tuple(_SEED + m for m in range(members)),
                collector=native, available_bytes=_available_bytes,
                initializer=initialize_member_winds, validation_callback=validation)
            assert report is not None and report["backend"] == "native_member_batched"
            assert report["executor"]["steps"] == 2
            assert report["executor"]["member_steps"] == 2 * members
            assert report["native_allocations"]["ordinary_member_forecasts_advanced"] == 0
            return report
        bootstrap_out = tmp_path / "native-bootstrap"
        bootstrap_out.mkdir()
        native_report = run_prepared_tree(inputs, output_directory=bootstrap_out,
            io_mode="history", initialization=WrfInitialization(inputs), ensemble_bootstrap=native_bootstrap)
        assert set(native_words.records) == set(control_words.records)
        assert _product_words(native_collector) == expected_products
        assert not list((tmp_path / "native-bootstrap").rglob("wrfout_*"))
        assert not list(native_collector.root.rglob("wrfout_*"))
        receipt["native_words"] = native_words.flush()
        receipt["native_forecast"] = native_report
        gc.collect()

    fallback_words = _WordCapture(tmp_path / "fallback-words", cfg,
                                  reference=control_words, members=1)
    fallback_collector = _collector(tmp_path / "fallback", inputs, members, renderer,
                                    keep=members == 1)
    fallback = _CapturedProducts(fallback_collector, fallback_words)
    def original_runner(shared_inputs, *, output_directory, **options):
        member = current_capture().member_id
        return _ordinary(shared_inputs, output_directory, member, fallback, try_native_n1=members == 1)
    session = PreparedEnsembleSession({"members": members, "keep_member_files": members == 1},
        output_directory=fallback_collector.root, collector=fallback, native_executor=None,
        identical_members="identity gate: every member must reproduce the ordinary forecast word for word")
    fallback_report = session.run_prepared(original_runner, inputs, io_mode="history")
    assert fallback_report["status"] == "PASS"
    assert fallback_report["members_completed"] == list(range(members))
    assert all(batch["execution_mode"] == "ordinary_member"
               for batch in fallback_report["packing"]["batches"])
    assert set(fallback_words.records) == set(control_words.records)
    assert _product_words(fallback_collector) == expected_products
    if members == 1:
        assert _history_words(fallback_collector.root / "members") == expected_history
        assert len(fallback_report["member_history_files"]) == len(expected_history)
    else:
        assert not fallback_report["member_history_files"]
        assert not list(fallback_collector.root.rglob("wrfout_*"))
    cp.cuda.get_current_stream().synchronize()
    receipt.update(status="PASS", fallback_words=fallback_words.flush(),
                   fallback_forecast=fallback_report, wall_seconds=time.perf_counter() - started,
                   identity="all state/physics/output words, aggregate CDF5 and Rust PNG bytes match")
    (tmp_path / "production-identity-receipt.json").write_text(
        json.dumps(_json_value(receipt), indent=2, sort_keys=True) + "\n")
