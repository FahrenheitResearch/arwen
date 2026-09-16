"""CPU-only remote review binds immutable scientific inputs before any launch."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm import remote_worker as rw


@pytest.fixture(autouse=True)
def cpu_only_memory_review(monkeypatch):
    monkeypatch.setattr("gpuwm.remote_plan.memory_review", lambda *_args, **_kwargs:
        {"measured": False, "free_bytes": None, "refuse": False, "warn": True,
         "verdict": "CPU-only snapshot contract test"})


def config(tmp_path):
    from gpuwm import domain_wizard as dw
    path = tmp_path / "science's 日本語.toml"
    text = dw.render_config(name="remote-snapshot", start_time=datetime(2026, 9, 5),
        hours=3, projection=dw._projection_entries(40, -100, "auto"),
        dims=dw._dims_for_scale(1, ()), ratios=(),
        fetch_hints=dict(source="gfs", cycle="2026-09-05T00", hours=3, out="data/cache", cadence=3), case_data=None)
    path.write_text(text, encoding="utf-8")
    return path


def request(tmp_path, source, **extra):
    return {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
            "config": str(source), "outdir": str(tmp_path / "new output"), "products": "none", **extra}


def test_dry_run_creates_nothing_and_binds_all_existing_companions(tmp_path):
    source = config(tmp_path)
    wps = source.with_suffix(".namelist.wps")
    wps.write_text("&share\n max_dom=1,\n/\n&geogrid\n geog_data_path='./geography',\n/\n", encoding="utf-8")
    companion = source.with_suffix(".d01-target.json")
    companion.write_text('{"nx": 32}\n', encoding="utf-8")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    payload = rw._launch(request(tmp_path, source, dry_run=True), tmp_path)
    review = payload["review"]
    assert payload["dry_run"] is True
    assert review["config_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert review["wps_sha256"] == hashlib.sha256(wps.read_bytes()).hexdigest()
    assert set(review["inputs"]) == {str(source), str(wps), str(companion)}
    assert review["argv"][4:7] == ["gpuwm.cli", "go", str(source)]
    assert "--no-memory-gate" in review["argv"]
    assert review["argv"][-2:] == ["--products", "none"]
    assert review["cwd"] == str(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("field", ["config", "wps", "input"])
def test_stale_review_refuses_before_output_or_job_creation(tmp_path, field):
    source = config(tmp_path)
    with pytest.raises(ValueError, match="changed since review"):
        rw._launch(request(tmp_path, source, **{f"expected_{field}_sha256": "0" * 64}), tmp_path)
    assert not (tmp_path / ".arwen-jobs").exists()
    assert not (tmp_path / "new output").exists()


def test_rebased_config_keeps_all_scientific_values_and_wps_path_meaning(tmp_path):
    source = config(tmp_path)
    wps = source.with_suffix(".namelist.wps")
    wps.write_text("&geogrid\n geog_data_path='./geo',\n opt_geogrid_tbl_path='../tables',\n/\n"
                   "&metgrid\n opt_metgrid_tbl_path='./met tables',\n/\n", encoding="utf-8")
    raw, originals, snapshots, binding = rw._inputs(source)
    assert tomllib.loads(snapshots[source.name].decode()) == raw
    assert originals[str(source)] == source.read_bytes()
    from gpuwm.namelist_import import parse_namelist_text
    relocated = parse_namelist_text(snapshots[wps.name].decode())
    assert relocated["geogrid"]["geog_data_path"] == [str((tmp_path / "geo").resolve())]
    assert relocated["geogrid"]["opt_geogrid_tbl_path"] == [str((tmp_path / "../tables").resolve())]
    assert relocated["metgrid"]["opt_metgrid_tbl_path"] == [str((tmp_path / "met tables").resolve())]
    assert binding["input_sha256"]


def test_relocation_preserves_declared_case_data_paths(tmp_path):
    from gpuwm.toml_document import emit_experiment_toml
    source = config(tmp_path)
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    raw["case_data"] = {"geog_root": "geography", "forcing": ["forcing.nc"], "vtable": "Vtable",
                        "wps_namelist": "inputs.wps", "sfcp_to_sfcp": True, "output_title": "fixture"}
    (tmp_path / "geography").mkdir()
    for name in ("forcing.nc", "Vtable", "inputs.wps"):
        (tmp_path / name).write_text("fixture")
    source.write_text(emit_experiment_toml(raw), encoding="utf-8")
    original, _, snapshots, _ = rw._inputs(source)
    actual = tomllib.loads(snapshots[source.name].decode())
    expected = dict(original)
    from gpuwm.case_data import resolved_case_data_paths
    expected["case_data"] = resolved_case_data_paths(original["case_data"], base_dir=tmp_path, source=str(source))
    expected["case_data"]["wps_namelist"] = "declared-inputs/namelist.wps"
    expected["case_data"]["vtable"] = "declared-inputs/Vtable"
    assert actual == expected
    assert snapshots["declared-inputs/Vtable"] == b"fixture"
    assert "declared-inputs/namelist.wps" in snapshots


def test_an_existing_case_folder_collects_this_run_without_modifying_what_it_holds(tmp_path):
    """An existing output directory is a case folder; the run claims a folder inside it."""
    from gpuwm import run_stamp
    source = config(tmp_path)
    output = tmp_path / "new output"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("existing run", encoding="utf-8")
    review = rw._launch({**request(tmp_path, source), "dry_run": True}, tmp_path)["review"]
    assert Path(review["run_root"]).parent == output and run_stamp.is_run_folder(review["run_root"])
    assert marker.read_text() == "existing run"
    assert not (tmp_path / ".arwen-jobs").exists()
    assert not Path(review["run_root"]).exists(), "a dry run claims nothing"


def test_an_existing_run_folder_is_refused_without_modifying_it(tmp_path):
    source = config(tmp_path)
    output = tmp_path / "run-20260905-000000Z"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("existing run", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        rw._launch(request(tmp_path, source, outdir=str(output)), tmp_path)
    assert marker.read_text() == "existing run"
    assert not (tmp_path / ".arwen-jobs").exists()


@pytest.mark.parametrize("identifier", ["../other", "/etc", "a/b", "", "x" * 129, ".", "id\n"])
def test_job_ids_are_confined_before_filesystem_lookup(tmp_path, identifier):
    with pytest.raises(ValueError, match="invalid remote job ID"):
        rw._directory(tmp_path, identifier)


def test_protocol_does_not_offer_arbitrary_command_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(rw.sys, "platform", "linux")
    with pytest.raises(ValueError, match="unsupported remote request fields"):
        rw.dispatch({"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
                     "argv": ["arbitrary-command"]})


def test_invalid_scientific_schema_fails_in_read_only_review(tmp_path):
    source = config(tmp_path)
    source.write_text(source.read_text(encoding="utf-8").replace("nx = ", "nx = -"), encoding="utf-8")
    with pytest.raises(ValueError):
        rw._launch(request(tmp_path, source, dry_run=True), tmp_path)
    assert not (tmp_path / ".arwen-jobs").exists()


def _restart(path, *, grid_id=1, domain_ids=None, corrupt=False, written_mode=None):
    import numpy as np
    array = np.arange(4, dtype=np.float32)
    header = {"format_version": 3, "grid_id": grid_id,
              "array_manifest": {"state/u": {"shape": [4], "dtype": "float32"}}}
    if domain_ids is not None:
        header["domain_ids"] = domain_ids
    if written_mode is not None:
        header["written_mode"] = written_mode
    arrays = {"state/u": array, "__gpuwm_restart_header__": np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)}
    if corrupt:
        arrays["undeclared"] = array
    with path.open("wb") as stream:
        np.savez(stream, **arrays)
    return path


def test_latest_restart_uses_only_complete_valid_sets_in_owned_output(tmp_path):
    output = tmp_path / "old output"
    forecast = output / "chain" / "run"
    forecast.mkdir(parents=True)
    source = config(tmp_path)
    old = {"outdir": str(output), "snapshot_config": str(source)}
    good = _restart(forecast / "gpuwmrst_d01_2026-09-05_01_00_00.npz")
    # A newer, torn tree is not a restart just because d01 is readable.
    torn = _restart(forecast / "gpuwmrst_d01_2026-09-05_02_00_00__tree.npz", domain_ids=[1, 2])
    assert rw._checkpoint(old, "latest") == good
    with pytest.raises(ValueError, match="torn set"):
        rw._checkpoint(old, str(torn))
    with pytest.raises(ValueError, match="source job"):
        rw._checkpoint(old, str(_restart(tmp_path / "elsewhere.npz")))


def test_restart_binding_covers_every_member_not_only_root(tmp_path):
    first = _restart(tmp_path / "gpuwmrst_d01_2026-09-05_01_00_00__tree.npz", domain_ids=[1, 2])
    second = _restart(tmp_path / "gpuwmrst_d02_2026-09-05_01_00_00__tree.npz", grid_id=2, domain_ids=[1, 2])
    before = rw._checkpoint_binding(first)
    assert set(before["checkpoint_inputs"]) == {str(first), str(second)}
    second.write_bytes(second.read_bytes() + b"changed sibling")
    after = rw._checkpoint_binding(first)
    assert after["checkpoint_sha256"] == before["checkpoint_sha256"]
    assert after["checkpoint_set_sha256"] != before["checkpoint_set_sha256"]


def test_resume_review_resolves_exact_checkpoint_and_preserves_old_output(tmp_path, monkeypatch):
    from gpuwm.toml_document import emit_experiment_toml
    source = config(tmp_path)
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    raw["case_data"] = {"geog_root": "geography", "forcing": ["forcing.nc"], "vtable": "Vtable",
                        "wps_namelist": "inputs.wps", "sfcp_to_sfcp": True, "output_title": "fixture"}
    (tmp_path / "geography").mkdir()
    for name in ("forcing.nc", "Vtable", "inputs.wps"):
        (tmp_path / name).write_text("fixture")
    source.write_text(emit_experiment_toml(raw), encoding="utf-8")
    output = tmp_path / "old output"
    output.mkdir()
    checkpoint = _restart(output / "gpuwmrst_d01_2026-09-05_01_00_00.npz")
    old = {"id": "old-job", "snapshot_config": str(source), "outdir": str(output), "cwd": str(tmp_path), "products": "t2"}
    monkeypatch.setattr(rw, "_directory", lambda workspace, job: tmp_path)
    monkeypatch.setattr(rw, "_record", lambda directory: old)
    monkeypatch.setattr(rw, "_status", lambda directory: {"state": "stopped"})
    wanted = request(tmp_path, source, action="resume", job="old-job", dry_run=True)
    wanted.pop("products")  # Omission inherits the saved run's products.
    review = rw._launch(wanted, tmp_path)["review"]
    assert review["checkpoint"] == str(checkpoint)
    assert review["parent_job"] == "old-job" and review["products"] == "t2"
    # A resume that does name one is asking for it, and gets it.
    named = rw._launch({**wanted, "products": "none"}, tmp_path)["review"]
    assert named["products"] == "none"
    assert review["argv"][-2:] == ["--restart", str(checkpoint)]
    assert not (tmp_path / "new output").exists()
    assert list(output.iterdir()) == [checkpoint]
    checkpoint.write_bytes(checkpoint.read_bytes() + b"different bytes, still readable")
    with pytest.raises(ValueError, match="checkpoint.*changed since review"):
        rw._launch({**wanted, "expected_checkpoint_set_sha256": review["checkpoint_set_sha256"]}, tmp_path)


@pytest.mark.parametrize("explicit", [False, True])
def test_a_resume_review_discloses_the_written_road_and_the_resolved_mode(tmp_path, monkeypatch, explicit):
    """The review carries what ``gpuwm resume`` prints, through the same door.

    The source job's configuration says which memory mode this resume
    resolves [tiles] to; the checkpoint's own stamp says which road wrote
    it.  Both are disclosures: the review binds the same checkpoint with
    or without them and never refuses on them.
    """
    from gpuwm.toml_document import emit_experiment_toml
    source = config(tmp_path)
    # Declared inputs, so the review has nothing to bind but the checkpoint.
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    raw["case_data"] = {"geog_root": "geography", "forcing": ["forcing.nc"], "vtable": "Vtable",
                        "wps_namelist": "inputs.wps", "sfcp_to_sfcp": True, "output_title": "fixture"}
    (tmp_path / "geography").mkdir()
    for name in ("forcing.nc", "Vtable", "inputs.wps"):
        (tmp_path / name).write_text("fixture")
    source.write_text(emit_experiment_toml(raw), encoding="utf-8")
    output = tmp_path / "old output"
    output.mkdir()
    stamped = _restart(output / "gpuwmrst_d01_2026-09-05_01_00_00.npz",
                       written_mode={"mode": "streamed", "shape": [4, 4], "store": "host"})
    old = {"id": "old-job", "snapshot_config": str(source), "outdir": str(output), "cwd": str(tmp_path), "products": "t2"}
    monkeypatch.setattr(rw, "_directory", lambda workspace, job: tmp_path)
    monkeypatch.setattr(rw, "_record", lambda directory: old)
    monkeypatch.setattr(rw, "_status", lambda directory: {"state": "stopped"})
    wanted = request(tmp_path, source, action="resume", job="old-job", dry_run=True)
    wanted.pop("products")
    if explicit:
        wanted["from_checkpoint"] = str(stamped)
    review = rw._launch(wanted, tmp_path)["review"]
    assert review["checkpoint"] == str(stamped)
    notes = review["resume_notes"]
    assert len(notes) == 2
    assert "this run resolves [tiles] to resident" in notes[0]
    assert "this checkpoint was WRITTEN streamed and this run resolves [tiles] to resident" in notes[1]
    assert all("mode-independent" in note for note in notes)
    # CONTROL: a file that names no road contributes nothing, and the same
    # review otherwise: one note from the configuration, the same binding.
    stamped.unlink()
    unstamped = _restart(output / "gpuwmrst_d01_2026-09-05_01_00_00.npz")
    if explicit:
        wanted["from_checkpoint"] = str(unstamped)
    plain = rw._launch(wanted, tmp_path)["review"]
    assert plain["checkpoint"] == str(unstamped)
    assert len(plain["resume_notes"]) == 1
    assert "this run resolves [tiles] to resident" in plain["resume_notes"][0]
    assert not (tmp_path / "new output").exists()


def test_a_fresh_start_review_carries_no_resume_notes(tmp_path):
    review = rw._launch(request(tmp_path, config(tmp_path), dry_run=True), tmp_path)["review"]
    assert review["resume_notes"] == []


def test_resume_without_a_written_checkpoint_keeps_its_public_refusal(tmp_path):
    """Renamed with the defect: "checkpointless" named a route that never existed.

    The refusal itself is unchanged and still correct, a source job that
    wrote no checkpoint cannot be resumed, but it is about a RUN that
    wrote nothing, not about a route that cannot write.
    """
    source = config(tmp_path)
    output = tmp_path / "old output"
    output.mkdir()
    with pytest.raises(ValueError, match="No manifest-valid checkpoint exists") as caught:
        rw._checkpoint({"outdir": str(output), "snapshot_config": str(source)}, "latest")
    assert "does not declare a valid forecast domain" not in str(caught.value)
    assert list(output.iterdir()) == []


def test_prepared_snapshot_preserves_bound_authority_bytes_exactly(tmp_path):
    source = config(tmp_path)
    source.write_bytes(b"# retained byte-bound comment\r\n" + source.read_bytes())
    wps = tmp_path / "exact remote namelist.wps"
    wps.write_bytes(b"! retained comment\r\n&geogrid\r\n geog_data_path='./geo',\r\n/\r\n")
    _, originals, snapshots, binding = rw._inputs(source, preserve_bound=True, extra_wps=wps)
    assert snapshots[source.name] == source.read_bytes()
    assert snapshots["prepared-inputs/namelist.wps"] == wps.read_bytes()
    assert originals[str(wps)] == wps.read_bytes()
    assert binding["wps_sha256"] == rw._file_sha(wps)


def test_prepared_receipt_review_binds_reused_bundle_and_refuses_drift(tmp_path):
    from gpuwm import stage_cli
    source = config(tmp_path)
    prepared = tmp_path / "existing prepared"
    prepared.mkdir()
    # This fixture exercises the public receipt/digest relay only. Real cache
    # preflight and forecast execution belong to the installed Linux fixture.
    schema = next(name for name, entry in stage_cli._schema_index().items() if entry["layout"] == "tree")
    receipt = prepared / "receipt.json"
    receipt.write_text(json.dumps({"schema": schema}), encoding="utf-8")
    wanted = request(tmp_path, source, prepared_root=str(prepared), dry_run=True)
    review = rw._launch(wanted, tmp_path)["review"]
    assert review["prepared_root"] == str(prepared)
    assert review["prepared_document"] == str(receipt)
    assert review["prepared_sha256"] == rw._file_sha(receipt)
    assert review["argv"][-2:] == ["--prepared-root", str(prepared)]
    assert "full content preflight at launch" in review["prepared_validation"]
    assert not (tmp_path / ".arwen-jobs").exists()
    receipt.write_text(json.dumps({"schema": schema, "changed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="prepared inputs changed since review"):
        rw._launch({**wanted, "expected_prepared_sha256": review["prepared_sha256"]}, tmp_path)


def test_explicit_wps_requires_prepared_bundle(tmp_path):
    source = config(tmp_path)
    with pytest.raises(ValueError, match="wps_namelist requires prepared_root"):
        rw._launch(request(tmp_path, source, wps_namelist=str(tmp_path / "namelist.wps"), dry_run=True), tmp_path)


@pytest.mark.parametrize("override", [False, True])
def test_resume_uses_explicit_inputs_or_saved_defaults(tmp_path, monkeypatch, override):
    source = config(tmp_path)
    output = tmp_path / "old-output"
    output.mkdir()
    checkpoint = _restart(output / "gpuwmrst_d01_2026-09-05_01_00_00.npz")
    old = {"id": "old-job", "snapshot_config": str(source),
           "outdir": str(output), "cwd": str(tmp_path), "products": "t2"}
    selected = {"products": "none"}
    for key in ("geog_root", "prepared_root"):
        before, after = tmp_path / ("old-" + key), tmp_path / ("new-" + key)
        before.mkdir()
        after.mkdir()
        old[key], selected[key] = str(before), str(after)
    old_wps, new_wps = tmp_path / "old.wps", tmp_path / "new.wps"
    old_wps.write_text("&share\n max_dom=1,\n/\n", encoding="utf-8")
    new_wps.write_text("&share\n max_dom=1,\n/\n! revised authority\n", encoding="utf-8")
    old["snapshot_wps_namelist"] = str(old_wps)
    selected["wps_namelist"] = str(new_wps)
    original = json.loads(json.dumps(old))
    checked = []
    monkeypatch.setattr(rw, "_directory", lambda workspace, job: tmp_path)
    monkeypatch.setattr(rw, "_record", lambda directory: old)
    monkeypatch.setattr(rw, "_status", lambda directory: {"state": "stopped"})
    def prepared_binding(prepared, config, wps, outdir):
        checked.append((str(prepared), str(wps)))
        return {"prepared_sha256": "a" * 64}
    monkeypatch.setattr(rw, "_prepared_binding", prepared_binding)
    wanted = request(tmp_path, source, action="resume", job="old-job", dry_run=True)
    wanted.pop("products")
    if override:
        wanted.update(selected)
    review = rw._launch(wanted, tmp_path)["review"]
    expected = selected if override else {**old, "wps_namelist": str(old_wps)}
    for name in selected:
        assert review[name] == expected[name]
        flag = "--" + name.replace("_", "-")
        assert review["argv"][review["argv"].index(flag) + 1] == expected[name]
    assert checked == [(expected["prepared_root"], expected["wps_namelist"])]
    assert review["checkpoint"] == str(checkpoint)
    assert old == original and list(output.iterdir()) == [checkpoint]
    assert not (tmp_path / "new output").exists()
