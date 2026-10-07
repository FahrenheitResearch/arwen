"""Native exporter command wiring and subprocess lifetime, without data processing."""
from datetime import datetime
from pathlib import Path
import subprocess
import threading

import pytest

from gpuwm import grib2_live, runplan


def test_standalone_and_live_doors_use_woof_definitions_and_grid_winds(tmp_path, monkeypatch, capsys):
    import argparse
    from gpuwm import grib2_export
    parser = argparse.ArgumentParser()
    grib2_export.register_cli(parser.add_subparsers())
    args = parser.parse_args(["export-grib2", "frame.nc", "--out", str(tmp_path)])
    request = grib2_export.build_request(args)
    live = grib2_live.request_for(["frame.nc"], tmp_path)
    assert request["definitions"] == live["definitions"] == "woof"
    assert request["winds"] == live["winds"] == "grid"
    # Spec 3.5 and D30: the GPU path is the default at both doors.
    assert request["post_device"] == live["post_device"] == "auto"
    # RULINGS 7, ASOS-scored: the mixed-layer TKE gust where the history
    # carries TKE, the similarity gust otherwise.
    assert request["gust"] == live["gust"] == "auto"
    # Restored options are off by default: no extrema, the SRW composite level.
    assert request["extrema_interval_seconds"] is live["extrema_interval_seconds"] is None
    assert request["composite_level_type"] == live["composite_level_type"] == 200
    assert "upp_control" not in request
    alternate = parser.parse_args(["export-grib2", "frame.nc", "--out", str(tmp_path),
                                   "--winds", "earth", "--post-device", "cpu", "--gust", "tke"])
    alternate_request = grib2_export.build_request(alternate)
    assert (alternate_request["winds"], alternate_request["post_device"], alternate_request["gust"]) == (
        "earth", "cpu", "tke")
    # `upp` is a one-release alias: same request, and the rename is said.
    alias = grib2_export.build_request(parser.parse_args(
        ["export-grib2", "frame.nc", "--out", str(tmp_path), "--definitions", "upp"]))
    assert alias == request
    assert "now woof" in capsys.readouterr().err
    # RULINGS 13: the three options 2.8.5-era requests named come back.
    for name in ("renderer", "arwen"):
        restored = grib2_export.build_request(parser.parse_args(
            ["export-grib2", "frame.nc", "--out", str(tmp_path), "--definitions", name,
             "--extrema-interval-seconds", "3600", "--upp-control", "rapr"]))
        assert restored["definitions"] == "renderer"
        assert restored["extrema_interval_seconds"] == 3600
        assert restored["composite_level_type"] == 10
        assert {k: v for k, v in restored.items()
                if k not in ("definitions", "extrema_interval_seconds", "composite_level_type")} == {
            k: v for k, v in request.items()
            if k not in ("definitions", "extrema_interval_seconds", "composite_level_type")}
    for bad in (["--extrema-interval-seconds", "0"], ["--extrema-interval-seconds", "86401"]):
        with pytest.raises(ValueError, match="extrema-interval-seconds"):
            grib2_export.build_request(parser.parse_args(["export-grib2", "frame.nc", "--out", str(tmp_path), *bad]))
    for bad in (["--definitions", "hrrr"], ["--upp-control", "nam"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["export-grib2", "frame.nc", "--out", str(tmp_path), *bad])
    from gpuwm import bridges
    from gpuwm import bridge_assets
    from types import SimpleNamespace
    assert bridges.BRIDGE_ABI_MARKERS["rw_grib2export"] == grib2_export.ABI_MARKER.encode()
    artifact, = [a for a in bridge_assets.BUNDLED_ARTIFACTS if a.name == "rw_grib2export"]
    assert (artifact.kind, artifact.crate, artifact.env_var) == (
        "executable", bridges.RUSTWX_CRATE_RELATIVE, grib2_export.BINARY_ENV)
    # The held-out UPP-derived exporter (2.8.5 and older) speaks the same
    # request schema but another catalog; it must never be taken for this one.
    held_out = ("rw_grib2export --request REQUEST.json schema=grib2-export.request/v1 "
                "modes=run,append,finalize progress=jsonl default_definitions=upp extrema_intervals=declared "
                "grid_geometry=wrf-native-locations/v1 upp_control=srw,rapr surface_catalog=upp-post-products/v6")
    # A 2.8.7 build from before the options came back is refused too: it
    # would refuse every request that names them.
    for stale in (held_out, grib2_export.ABI_MARKER.split(" post_device=")[0],
                  grib2_export.ABI_MARKER.split(" definitions=")[0]):
        monkeypatch.setattr(grib2_export.subprocess, "run", lambda *a, marker=stale, **k:
            SimpleNamespace(returncode=0, stdout=marker))
        assert not grib2_export.BINARY.probe(Path("stale-exporter"))[0]
    monkeypatch.setattr(grib2_export.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=0, stdout=grib2_export.ABI_MARKER))
    assert grib2_export.BINARY.probe(Path("woof-exporter"))[0]


def test_rust_abi_literal_matches_the_python_marker():
    """rw_grib2export::ABI and the Python marker are one literal."""
    import re
    from gpuwm import grib2_export
    source = (Path(__file__).resolve().parents[1] / "tools" / "rustwx" / "crates" / "rw-grib2export"
              / "src" / "lib.rs").read_text(encoding="utf-8")
    match = re.search(r'pub const ABI: &str = "(.*?)";', source, re.S)
    assert match, "rw_grib2export::ABI not found"
    literal = re.sub(r"\\\n\s*", "", match.group(1))
    assert literal == grib2_export.ABI_MARKER


def test_default_standalone_request_is_the_live_request(tmp_path):
    import argparse
    from gpuwm import grib2_export

    parser = argparse.ArgumentParser()
    grib2_export.register_cli(parser.add_subparsers())
    default = grib2_export.build_request(parser.parse_args(["export-grib2", "frame.nc", "--out", str(tmp_path)]))
    assert default == grib2_live.request_for(["frame.nc"], tmp_path)


def test_invoke_leaves_the_forecast_card_visible(tmp_path, monkeypatch):
    """The post runs on a GPU by default, so the door must not hide the cards."""
    from gpuwm import grib2_export
    seen = {}

    class Process:
        def __init__(self, argv, **kwargs):
            seen["env"] = kwargs["env"]
            import io
            self.stdout = io.StringIO('{"event": "done", "frames": 0, "bytes": 0, "seconds": 0.0}\n')
        def poll(self):
            return 0
        def wait(self, timeout=None):
            return 0

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(grib2_export.subprocess, "Popen", Process)
    assert grib2_export.invoke(Path("native"), {"schema": grib2_export.REQUEST_SCHEMA}) == 0
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert seen["env"].get("GPUWM_NO_LOCAL_GPU") == __import__("os").environ.get("GPUWM_NO_LOCAL_GPU")


def test_queue_is_nonblocking_deduplicated_serial_and_finalizes(tmp_path):
    started = threading.Event()
    release = threading.Event()
    requests = []

    def invoke(executable, request, **kwargs):
        requests.append(request)
        assert kwargs["as_json"] is True
        assert callable(kwargs["on_process"])
        if len(requests) == 1:
            started.set()
            assert release.wait(5)
        return 0

    export = grib2_live.LiveGrib2Export(tmp_path / "grib2", executable=Path("native"), invoke=invoke)
    first = tmp_path / "first.nc"
    later = tmp_path / "later.nc"
    assert export.frame_committed(domain=1, valid_time=datetime(2024, 5, 25, 18), path=first)
    assert started.wait(5)
    # The second writer callback returns while the first native process waits.
    assert export.frame_committed(domain=2, valid_time="2024-05-25T19:00:00", path=later)
    assert not export.frame_committed(domain=1, valid_time="ignored", path=first)
    release.set()
    summary = export.stop()
    assert [r["mode"] for r in requests] == ["append", "append", "finalize"]
    assert requests[0]["inputs"] == [str(first.resolve())]
    assert requests[1]["inputs"] == [str(later.resolve())]
    assert requests[2]["inputs"] == [] and requests[2]["zip"] is True
    assert summary["completed_frames"] == 2
    assert summary["finalized"] and summary["ended"]
    assert export.stop() == summary
    assert not export.frame_committed(domain=1, valid_time="late", path=tmp_path / "late.nc")


def test_export_failure_is_reported_and_never_finalizes(tmp_path):
    warnings = []
    requests = []

    def invoke(executable, request, **kwargs):
        requests.append(request)
        return 9

    export = grib2_live.LiveGrib2Export(tmp_path / "grib2", executable=Path("native"), invoke=invoke,
        warn=lambda code, message, **fields: warnings.append((code, message, fields)))
    export.frame_committed(domain=1, valid_time="2024-05-25T18:00:00", path=tmp_path / "frame.nc")
    with pytest.raises(RuntimeError, match="GRIB2 export incomplete.*exited 9"):
        export.stop()
    assert [r["mode"] for r in requests] == ["append"]
    assert warnings[0][0] == "grib2_export_failed"
    assert not export.summary()["finalized"]


def test_context_is_scanned_once_per_parent_and_excludes_archived_attempts(tmp_path):
    requests = []
    parent = tmp_path / "history"
    parent.mkdir()
    earlier = parent / "wrfout_d01_2024-05-25_18-00-00"
    current = parent / "wrfout_d01_2024-05-25_19-00-00.nc"
    for path in (earlier, current, parent / "wrfout_d01.tmp.nc",
                 parent / "wrfout_d01.nc.part", parent / "wrfout_d01.nc.gz"):
        path.write_bytes(b"names only")
    archived = parent / "previous-attempt"
    archived.mkdir()
    (archived / "wrfout_d01_2024-05-25_17-00-00.nc").write_bytes(b"names only")
    export = grib2_live.LiveGrib2Export(tmp_path / "grib2", executable=Path("native"),
        invoke=lambda exe, request, **kwargs: requests.append(request) or 0)
    export.frame_committed(domain=1, valid_time="2024-05-25T19:00:00", path=current)
    export.frame_committed(domain=1, valid_time="2024-05-25T20:00:00", path=parent / "later.nc")
    export.stop()
    assert requests[0]["inputs"] == [str(earlier.resolve()), str(current.resolve())]
    assert requests[1]["inputs"] == [str((parent / "later.nc").resolve())]


def test_halt_ends_explicit_owned_process_and_discards_queue(tmp_path):
    started = threading.Event()
    processes = []
    requests = []

    def invoke(executable, request, **kwargs):
        import sys
        requests.append(request)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        processes.append(child)
        kwargs["on_process"](child)
        started.set()
        code = child.wait()
        kwargs["on_process"](None)
        return code

    export = grib2_live.LiveGrib2Export(tmp_path / "grib2", executable=Path("native"), invoke=invoke)
    export.frame_committed(domain=1, valid_time="2024-05-25T18:00:00", path=tmp_path / "a.nc")
    assert started.wait(5)
    export.frame_committed(domain=1, valid_time="2024-05-25T19:00:00", path=tmp_path / "b.nc")
    summary = export.halt()
    assert processes[0].poll() is not None
    assert len(requests) == 1
    assert summary["halted"] and summary["ended"] and not summary["finalized"]
    assert not summary["errors"]


def test_observer_dispatches_grib2_without_maps_and_closes_lifecycle(tmp_path, monkeypatch):
    requests = []

    class Export:
        def __init__(self, out, **kwargs):
            requests.append(("arm", out))
        def frame_committed(self, **kwargs):
            requests.append(("frame", kwargs))
        def stop(self):
            requests.append(("stop", None))
            return {"finalized": True}
        def halt(self):
            requests.append(("halt", None))
            return {"halted": True}

    class Events:
        def emit(self, event, **fields):
            pass

    monkeypatch.setattr(grib2_live, "LiveGrib2Export", Export)
    observer = runplan.RunObserver(Events())
    observer.arm_grib2(tmp_path / "grib2")
    assert observer.live_products is None and observer.first_products is None
    observer.output_committed(domain=2, valid_time="2024-05-25T18:00:00", path=tmp_path / "frame.nc")
    observer.complete(3600)
    assert [r[0] for r in requests] == ["arm", "frame", "stop"]
    observer.stop_grib2(halt=True)
    assert requests[-1][0] == "halt"
    observer.restarting("recovery")
    assert requests[-1] == ("arm", tmp_path / "grib2-attempt-02")
    observer.restarting("recovery")
    assert requests[-1] == ("arm", tmp_path / "grib2-attempt-03")


def test_runplan_option_and_cli_flags_are_reachable(tmp_path):
    import argparse
    from gpuwm import go_cli, render
    assert runplan._run_option("grib2", True, tmp_path) is True
    assert runplan._run_option("grib2", False, tmp_path) is False
    with pytest.raises(runplan.PlanError, match="must be true or false"):
        runplan._run_option("grib2", "true", tmp_path)
    assert "grib2" in runplan.ROUTES["prepared"].run_options
    assert "grib2" in runplan.ROUTES["experiment"].run_options
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    render.register_cli(sub)
    go_cli.register_cli(sub)
    assert parser.parse_args(["render", "frame.nc", "--grib2-out", str(tmp_path)]).grib2_out == tmp_path
    assert parser.parse_args(["go", "config.toml", "--grib2", "--products", "none"]).grib2 is True
    from gpuwm.ensemble.request import EnsembleRequest
    grib2_live.validate_member_contract(True, None)
    grib2_live.validate_member_contract(True, EnsembleRequest(members=1))
    grib2_live.validate_member_contract(False, EnsembleRequest(members=2))


def test_go_grib2_prepared_route_uses_registered_observer(monkeypatch, tmp_path):
    from gpuwm import go_cli
    from types import SimpleNamespace
    config = tmp_path / "config.toml"
    config.write_text('[fetch]\nsource="gfs"\ncycle="2024-05-25T18"\n')
    args = SimpleNamespace(config=config, readiness=False, no_probe=False,
        prepared_root=None, restart=None, wps_namelist=None, supplement=None,
        data_dir=None, grib2=True, cycle=None)
    monkeypatch.setattr(go_cli, "_flag_cycle", lambda *a: None)
    monkeypatch.setattr(go_cli, "_extend_outdir", lambda *a: None)
    monkeypatch.setattr(runplan, "prepared_chain_for_source", lambda *a, **k: "prepared:go")
    monkeypatch.setattr(go_cli, "_registered_launch", lambda *a, **k: 37)
    monkeypatch.setattr(go_cli, "_go_prepared_main", lambda *a, **k: pytest.fail("second observer"))
    assert go_cli.go_main(args) == 37


@pytest.mark.parametrize("recipe", [None, "time-lagged"])
def test_go_ensemble_grib2_refuses_before_recipe_planning(recipe, tmp_path, monkeypatch, capsys):
    from gpuwm import capabilities, cli, go_cli, provenance_gate
    from gpuwm.ensemble import recipe_door
    config = tmp_path / "members.toml"
    text = '[fetch]\nsource="gfs"\ncycle="2026-10-02T17"\n[ensemble]\nmembers=2\n'
    if recipe is not None:
        text += f'recipe="{recipe}"\n'
    config.write_text(text)
    monkeypatch.setattr(provenance_gate, "announce", lambda *a, **k: None)
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)
    monkeypatch.setattr(recipe_door, "plan_recipe", lambda *a, **k: pytest.fail("planned before refusal"))
    assert cli.main(["go", str(config), "--grib2"]) == 2
    assert "domain/time export identity would collide" in capsys.readouterr().err
    # The same ensemble without live export still reaches its ordinary route.
    monkeypatch.setattr(go_cli, "_go_recipe", lambda *a, **k: 37)
    assert cli.main(["go", str(config)]) == 37


@pytest.mark.parametrize("source", ["config", "run_options"])
def test_runplan_ensemble_grib2_refuses_before_preparation(source, tmp_path, monkeypatch, capsys):
    import json
    from gpuwm import capabilities, cli, provenance_gate
    config = tmp_path / "members.toml"
    config.write_text('[fetch]\nsource="gfs"\ncycle="2026-10-02T17"\n'
                      + ('[ensemble]\nmembers=2\n' if source == "config" else ''))
    options = {"grib2": True}
    if source == "run_options":
        options["ensemble"] = {"members": 2}
    document = {"schema": runplan.PLAN_SCHEMA, "name": "member-export", "route": "experiment",
                "config": {"path": str(config)}, "output_root": str(tmp_path / "run"),
                "run_options": options}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(document))
    monkeypatch.setattr(provenance_gate, "announce", lambda *a, **k: None)
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)
    monkeypatch.setattr(runplan, "resolve_plan", lambda *a, **k: pytest.fail("prepared before refusal"))
    assert cli.main(["run-plan", str(plan)]) == 2
    assert "domain/time export identity would collide" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()
    document["run_options"]["grib2"] = False
    runplan.build_plan(document, source="fixture", base_dir=tmp_path, sha256="0" * 64)
    document["run_options"]["grib2"] = True
    document["run_options"]["ensemble"] = {"members": 1}
    runplan.build_plan(document, source="fixture", base_dir=tmp_path, sha256="0" * 64)
