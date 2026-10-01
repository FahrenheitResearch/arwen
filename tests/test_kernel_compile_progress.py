"""The first run's kernel compile is visible: the door that pays it and the
event that shows it.

`gpuwm warm-kernels` compiles the forecast kernels into the kernel cache
ahead of the first forecast; the loader publishes each module it really
compiles, and `gpuwm run-plan` relays those as ``kernel_compile_progress``
warnings a page can show.  These tests hold the CPU half; the door itself
was run on a card (see CHANGELOG.md 2.8.0).
"""

from __future__ import annotations

import json
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
    # Every default a plain run can bind: each source's own, and the
    # suite each grid-spacing row gives a source that admits it, since a
    # plain sub-km forecast binds that row.
    expected = {physics_menu.default_profile_for(source)
                for source in physics_menu.registered_sources()} - {None}
    expected |= {physics_menu.default_profile_for(source, row["finest_dx_below_m"] / 2)
                 for source in physics_menu.registered_sources()
                 for row in physics_menu.SPACING_DEFAULTS} - {None}
    assert set(profiles) == expected
    assert {row["profile_id"] for row in physics_menu.SPACING_DEFAULTS} <= set(profiles)


# ---------------------------------------------------------------------------
# --all-profiles: every shipped suite warms, and the exit code means what
# failed.  The card half was run on an RTX 5090: 28 of 28 profiles
# warmed, exit 0, where 2.8.0 exited 1 at the first suite with no
# longwave scheme.  These hold the CPU half with stand-ins for the device.


def _stand_in_cupy(monkeypatch):
    import sys
    import types

    runtime = types.SimpleNamespace(
        deviceSynchronize=lambda: None,
        getDeviceProperties=lambda device: {"name": b"stand-in card"})
    cuda = types.SimpleNamespace(runtime=runtime,
                                 Device=lambda *a: types.SimpleNamespace(id=0))
    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(cuda=cuda))


def test_every_shipped_profile_hands_initialize_physics_a_longwave_it_accepts(
        monkeypatch):
    """The 13 suites with no longwave scheme were refused on 2.8.0.

    Their synthetic column had no downward longwave source, and
    initialize_physics refuses to invent one ("downward longwave (GLW)
    has no source"), so `gpuwm warm-kernels --all-profiles` stopped at
    the first of them.  The stand-in below runs the ENGINE'S OWN guard
    over exactly what the door hands initialize_physics.
    """

    from gpuwm import physics_menu, warm_kernels
    from gpuwm.config import radiation_scheme_ids
    from gpuwm.core import dycore, grid, moist
    from gpuwm.core import physics as core_physics
    from gpuwm.physics_compat import DECLARED_CONSTANT_GLW_WM2

    _stand_in_cupy(monkeypatch)
    monkeypatch.setattr(grid, "make_vertical_coord", lambda *a, **k: "coord")
    monkeypatch.setattr(grid, "make_base_state", lambda *a, **k: "base")
    monkeypatch.setattr(moist, "init_moist_balanced", lambda *a, **k: "state")
    monkeypatch.setattr(dycore, "step", lambda state, cfg: None)
    handed = []

    def initialize_physics(state, cfg, **kwargs):
        longwave, shortwave = radiation_scheme_ids(cfg)
        glw, provenance = core_physics._resolve_initial_glw(
            kwargs.get("glw"), ra_lw_physics=longwave,
            radiation_active=bool(longwave or shortwave),
            sf_surface_physics=int(cfg.sf_surface_physics))
        handed.append((kwargs.get("glw"), provenance, longwave))

    monkeypatch.setattr(core_physics, "initialize_physics", initialize_physics)

    declared = 0
    for profile in physics_menu.shipped_profiles():
        handed.clear()
        said = warm_kernels._run_profile(profile,
                                         warm_kernels.DEFAULT_LEVELS)
        (glw, provenance, longwave), = handed
        if longwave:
            # A suite with a longwave scheme is handed nothing: its
            # scheme writes GLW, and a number here would only pre-fill it.
            assert glw is None and provenance == "scheme", profile
        elif provenance == "declared":
            assert glw == DECLARED_CONSTANT_GLW_WM2, profile
            assert "declared constant" in said, profile
            declared += 1
    assert declared == 13


def test_all_profiles_exits_by_what_failed_and_runs_past_it(
        monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    from gpuwm import kernel_compile_notice as notice
    from gpuwm import physics_menu, warm_kernels
    from gpuwm.cli import build_parser

    _stand_in_cupy(monkeypatch)
    monkeypatch.setattr(notice, "current_compute_capability", lambda: "120")
    monkeypatch.setattr(notice, "cupy_kernel_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(notice, "scan_kernel_cache", lambda cache: (0, 0))
    monkeypatch.setattr(notice, "kernel_cache_state",
                        lambda cache, **k: SimpleNamespace(
                            entries_for_capability=0))
    profiles = physics_menu.shipped_profiles()
    ran = []

    def run(profile, levels):
        ran.append(profile)
        return "computed by ra_lw_physics=4"

    monkeypatch.setattr(warm_kernels, "_run_profile", run)
    args = build_parser().parse_args(["warm-kernels", "--all-profiles", "--json"])
    assert warm_kernels.warm_kernels_main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["failed"] == []
    assert [row["profile"] for row in report["profiles"]] == list(profiles)

    refused = profiles[1]

    def run_one_refused(profile, levels):
        ran.append(profile)
        if profile == refused:
            raise ValueError("stand-in refusal of this suite")
        return "computed by ra_lw_physics=4"

    ran.clear()
    monkeypatch.setattr(warm_kernels, "_run_profile", run_one_refused)
    assert warm_kernels.warm_kernels_main(args) == 1
    report = json.loads(capsys.readouterr().out)
    # The pass went on past the refused suite and named it.
    assert ran == list(profiles)
    assert report["failed"] == [{"profile": refused,
                                 "error": "stand-in refusal of this suite"}]
    assert len(report["profiles"]) == len(profiles) - 1


def test_a_profile_reading_an_unstaged_table_is_named_and_the_pass_goes_on(
        monkeypatch, tmp_path, capsys):
    """An install that has not run `gpuwm fetch-tables` has no Thompson
    tables, and the loader raises FileNotFoundError.  That stopped the
    whole pass with a traceback at the first Thompson suite; it is a
    property of the suites that read the table, not of the card."""

    from types import SimpleNamespace

    from gpuwm import kernel_compile_notice as notice
    from gpuwm import physics_menu, warm_kernels
    from gpuwm.cli import build_parser

    _stand_in_cupy(monkeypatch)
    monkeypatch.setattr(notice, "current_compute_capability", lambda: "120")
    monkeypatch.setattr(notice, "cupy_kernel_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(notice, "scan_kernel_cache", lambda cache: (0, 0))
    monkeypatch.setattr(notice, "kernel_cache_state",
                        lambda cache, **k: SimpleNamespace(
                            entries_for_capability=0))
    profiles = physics_menu.shipped_profiles()
    unstaged = profiles[0]
    ran = []

    def run(profile, levels):
        ran.append(profile)
        if profile == unstaged:
            raise FileNotFoundError(
                "missing Thompson table asset /stand-in/qr_acr_qg_V4.dat")
        return "computed by ra_lw_physics=4"

    monkeypatch.setattr(warm_kernels, "_run_profile", run)
    args = build_parser().parse_args(["warm-kernels", "--all-profiles", "--json"])
    assert warm_kernels.warm_kernels_main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert ran == list(profiles)
    assert report["failed"] == [{
        "profile": unstaged,
        "error": "missing Thompson table asset /stand-in/qr_acr_qg_V4.dat",
        "remedy": warm_kernels.MISSING_ASSET_REMEDY}]
    assert "gpuwm fetch-tables" in warm_kernels.MISSING_ASSET_REMEDY
    assert len(report["profiles"]) == len(profiles) - 1
