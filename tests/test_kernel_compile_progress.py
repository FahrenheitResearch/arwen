"""The first run's kernel compile is visible: the door that pays it and the
event that shows it.

`gpuwm warm-kernels` compiles the forecast kernels into the kernel cache
ahead of the first forecast; the loader publishes each module it really
compiles, and `gpuwm run-plan` relays those as ``kernel_compile_progress``
warnings a page can show.  These tests hold the CPU half; the door itself
was run on a card (see CHANGELOG.md 2.8.0).
"""

from __future__ import annotations

from pathlib import Path

from gpuwm import kernel_compile_notice as notice
from gpuwm import progress


def test_nothing_is_measured_or_published_when_no_run_is_watching(
        tmp_path, monkeypatch):
    monkeypatch.setenv(notice.CUPY_CACHE_ENV, str(tmp_path))
    assert not progress.event_sinks_installed()
    with notice.observe_module_compile("gpuwm.core.kernels:any"):
        (tmp_path / "entry").write_bytes(b"x")


def test_a_compile_is_published_and_a_cache_load_is_not(tmp_path,
                                                        monkeypatch):
    monkeypatch.setenv(notice.CUPY_CACHE_ENV, str(tmp_path))
    seen = []
    with progress.event_sink(lambda event, **fields: seen.append(
            (event, fields))):
        with notice.observe_module_compile("gpuwm.core.kernels:first"):
            (tmp_path / "compiled-1").write_bytes(b"x")
        with notice.observe_module_compile("gpuwm.core.kernels:cached"):
            pass
        with notice.observe_module_compile("gpuwm.core.kernels:second"):
            (tmp_path / "compiled-2").write_bytes(b"x")
            (tmp_path / "compiled-3").write_bytes(b"x")
    assert [event for event, _ in seen] == ["warning", "warning"]
    first, second = (fields for _, fields in seen)
    assert first["code"] == notice.COMPILE_PROGRESS_CODE
    assert first["message"] == "compiling GPU kernels"
    assert first["module"] == "gpuwm.core.kernels:first"
    assert first["cache_entries_written"] == 1
    assert second["module"] == "gpuwm.core.kernels:second"
    assert second["cache_entries_written"] == 2
    assert second["modules_compiled"] == first["modules_compiled"] + 1
    assert second["compile_seconds"] >= first["compile_seconds"]


def test_run_plan_relays_the_compile_as_a_declared_warning(tmp_path,
                                                          monkeypatch):
    from gpuwm import runplan

    assert notice.COMPILE_PROGRESS_CODE in runplan.WARNING_CODES
    monkeypatch.setenv(notice.CUPY_CACHE_ENV, str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()
    stream = runplan.EventStream(tmp_path / "events.jsonl", mirror=None)
    observer = runplan.RunObserver(stream)
    observer.enter_stage("initialize")
    with runplan._kernel_compile_relay(observer):
        with notice.observe_module_compile("gpuwm.core.kernels:ysu"):
            (tmp_path / "cache" / "entry").write_bytes(b"x")
        progress.emit_event("warning", code="preparation_progress",
                            message="not this relay's")
    stream.close()
    records = [record for record in runplan.read_events(
        tmp_path / "events.jsonl") if record["event"] == "warning"]
    assert len(records) == 1, records
    record = records[0]
    assert record["code"] == notice.COMPILE_PROGRESS_CODE
    assert record["message"] == "compiling GPU kernels"
    assert record["stage"] == "initialize"
    assert record["module"] == "gpuwm.core.kernels:ysu"


def test_the_loader_compiles_through_the_observer():
    """Both loader sites compile through it, so every get_kernel is seen."""

    source = (Path(__file__).resolve().parents[1] / "gpuwm" / "core"
              / "kernels" / "__init__.py").read_text(encoding="utf-8")
    assert source.count("_compile_observed(mod, ") == 2
    assert "mod.compile()" not in source


def test_warm_kernels_is_a_door_and_defaults_to_the_sources_defaults():
    from gpuwm import physics_menu
    from gpuwm.cli import build_parser
    from gpuwm.warm_kernels import default_profiles, warm_kernels_main

    args = build_parser().parse_args(["warm-kernels", "--json"])
    assert args.func is warm_kernels_main
    profiles = default_profiles()
    assert len(profiles) == len(set(profiles)) >= 1
    assert set(profiles) == {
        physics_menu.default_profile_for(source)
        for source in physics_menu.registered_sources()} - {None}
