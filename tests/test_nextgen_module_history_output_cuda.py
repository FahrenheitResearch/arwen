"""Repeated root-CLI diagnostics preserve prior files; not a physics replay."""
import hashlib
import json
import sys
import time
from types import ModuleType

import pytest

pytestmark = pytest.mark.gpu


def test_module_history_repeated_cli_preserves_completed_output(tmp_path, monkeypatch, capsys):
    import cupy as cp
    from gpuwm.cli import main
    from gpuwm.hydro import module_history, output

    factory = ModuleType("gpuwm._nextgen_history_output_probe")
    prepared, writers, receipts, timings = [], [], [], []

    class Source:
        def __init__(self):
            self.arrays = {
                "f32": cp.asarray([-0.0, 1.25], dtype=cp.float32),
                "f64": cp.asarray([-0.0, 1.125], dtype=cp.float64),
                "i32": cp.asarray([-7, 11], dtype=cp.int32),
                "i64": cp.asarray([-(2**40), 2**40], dtype=cp.int64),
            }
            self.restart_identity = {"scope": "typed output instrumentation; no runoff physics"}
            prepared.append(self)

        def advance(self, dt_seconds, *, stream=None):
            pass

    factory.prepare = lambda **kwargs: Source()
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    original = module_history.prepare_writer
    late = []

    def prepare_writer(**kwargs):
        started = time.perf_counter()
        writer = original(**kwargs)
        timings.append({"prepare_seconds": time.perf_counter() - started})
        writers.append(writer)
        if late:
            writer.outdir.mkdir()
            (writer.outdir / "arrived.txt").write_bytes(b"external file arrived after preparation\n")
        return writer

    monkeypatch.setattr(module_history, "prepare_writer", prepare_writer)
    config = tmp_path / "hydro.toml"

    def run(outdir, capacity, origin):
        text = f'''[hydro]
enabled = true
interval_seconds = 60
output_factory = "gpuwm.hydro.module_history:prepare_writer"
[[hydro.stages]]
name = "source"
factory = "{factory.__name__}:prepare"
[[hydro.stages]]
name = "history"
factory = "gpuwm.hydro.module_history:prepare"
[hydro.stages.options]
capacity_steps = {capacity}
catchment_ids = ["output-probe-a", "output-probe-b"]
origin = "{origin}"
'''
        for name in ("f32", "f64", "i32", "i64"):
            text += f'''[[hydro.stages.options.fields]]
component = "source"
field = "{name}"
name = "{name}"
units = "1"
'''
        config.write_text(text)
        started = time.perf_counter()
        assert main(["hydro", "run", str(config), "--seconds", str(capacity * 60),
                     "--outdir", str(outdir)]) == 0
        timings[-1]["root_cli_seconds"] = time.perf_counter() - started
        writer = writers[-1]
        saved = json.loads((writer.outdir / "receipt.json").read_text())
        receipts.append((writer, outdir, saved))
        timings[-1]["receipt"] = saved
        # Actual Rust reader returns native bytes; no Python numeric decode.
        metadata, payload = output._call({"operation": "dataset-read",
                                          "path": str(writer.outdir / "history.nc")})
        assert {d["name"]: d["len"] for d in metadata["schema"]["dimensions"]} == {"time": capacity, "catchment": 2}
        attrs = {a["name"]: a for a in metadata["schema"]["attributes"]}
        assert json.loads(attrs["history_identity"]["text"])["origin"] == origin
        fields = {v["name"]: v for v in metadata["schema"]["variables"]}
        for key, source in prepared[-1].arrays.items():
            variable = fields[key]
            expected = source.get().tobytes() * capacity
            start = variable["offset"]
            assert payload[start:start + len(expected)] == expected
        return writer, saved

    target = tmp_path / "completed"
    first, first_receipt = run(target, 2, "2023-04-02T00:00:00Z")
    assert first.outdir == target
    original_bytes = {p.name: p.read_bytes() for p in target.iterdir()}
    second, second_receipt = run(target, 3, "2023-04-03T00:00:00Z")
    assert second.outdir != target and second.outdir.parent == target.parent
    assert second.outdir.name.startswith("hydro-run-")
    assert first_receipt["steps"] == 2 and second_receipt["steps"] == 3
    assert {p.name: p.read_bytes() for p in target.iterdir()} == original_bytes
    assert hashlib.sha256((second.outdir / "history.nc").read_bytes()).digest() != hashlib.sha256(original_bytes["history.nc"]).digest()

    late.append(True)
    late_target = tmp_path / "late"
    third, _ = run(late_target, 2, "2023-04-04T00:00:00Z")
    assert third.outdir != late_target and third.outdir.parent == late_target.parent
    assert sorted(p.name for p in late_target.iterdir()) == ["arrived.txt"]
    assert (late_target / "arrived.txt").read_bytes() == b"external file arrived after preparation\n"
    for writer, requested, saved in receipts:
        assert saved["output_directory"] == str(writer.outdir)
        assert saved["requested_output_directory"] == str(requested)
    assert "Hydro output:" in capsys.readouterr().err
    # The separate restart.nc is resumable; a diagnostic archive still fails
    # the Rust checkpoint envelope validation before any driver mutation.
    with pytest.raises(RuntimeError, match="checkpoint"):
        third.restore(target / "history.nc", None)

    (tmp_path / "regression-receipt.json").write_text(json.dumps({
        "scope": "real CLI/GPU capture/Rust output; synthetic source, not physics",
        "card": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "timings": timings, "all_native_field_bytes_equal": True,
        "previous_files_unchanged": True, "late_path_preserved": True}, indent=2) + "\n")
