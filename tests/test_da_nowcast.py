"""The nowcast front door's argument surface and planning math.

Pure tests: no network, no GPU, no subprocess execution.  Sites in
these tests are synthetic ids -- real station names never enter the
tree's generic code or its fixtures (standing owner rule).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.da_nowcast import (
    STAGES, VERIFY_SCHEMA, WindowPlan, _stop, advance_state,
    build_parser, cycle_cmd, fetch_cmd, gallery_rows, geojson_box,
    handoff_state, initial_frames, latest_gfs_cycle,
    merge_gallery_rows, motion_from_centroids, obs_cmd, plan_window,
    resolve_latest_window_end, site_domain_center, start_verification,
    validate_site, verdict_line, verification_block, verify_obs_name,
    verify_pass, watch_cmd, wizard_cmd, wrfout_name, write_verification)
from tools.da_nowcast_render import footer_band_fraction


def utc(*parts) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


NOW = utc(2026, 8, 5, 5, 45)


# ---------------------------------------------------------------------------
# site ids are arguments, validated, never defaulted
# ---------------------------------------------------------------------------
class TestSiteValidation:
    def test_uppercases(self):
        assert validate_site("qqqq") == "QQQQ"

    def test_accepts_digit_tail(self) -> None:
        assert validate_site("Q9Z1") == "Q9Z1"

    @pytest.mark.parametrize("bad", ["", "QQ", "QQQQQ", "1QQQ",
                                     "QQ-Q", "QQQ "])
    def test_refuses_non_station_shapes(self, bad):
        with pytest.raises(SystemExit):
            validate_site(bad)

    def test_site_has_no_default(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--window-end", "latest",
                               "--out", "x"])

    def test_no_real_site_tokens_in_the_tools(self):
        """No radar-site names in generic code, defaults, or
        identifiers -- the standing owner rule, checked mechanically
        for the two names the orders used."""

        tools = Path(__file__).resolve().parent.parent / "tools"
        for name in ("da_nowcast.py", "da_nowcast_render.py"):
            source = (tools / name).read_text(encoding="utf-8")
            for token in ("KDMX", "KBMX"):
                assert token not in source, (name, token)
            assert not re.search(r'default\s*=\s*"[A-Z][A-Z0-9]{3}"',
                                 source), name


# ---------------------------------------------------------------------------
# window planning
# ---------------------------------------------------------------------------
class TestWindowPlan:
    def plan(self, **overrides) -> WindowPlan:
        kwargs = dict(cycles=6, cycle_seconds=900, free_legs=6,
                      now=NOW)
        kwargs.update(overrides)
        return plan_window(utc(2026, 8, 5, 5, 30), **kwargs)

    def test_init_lands_on_the_hour(self):
        plan = self.plan()
        assert plan.init == utc(2026, 8, 5, 4, 0)

    def test_cycle_times_span_the_window(self):
        plan = self.plan()
        assert plan.cycle_times[0] == utc(2026, 8, 5, 4, 15)
        assert plan.cycle_times[-1] == plan.window_end

    def test_free_leg_times_continue_the_cadence(self):
        plan = self.plan()
        assert plan.free_leg_times()[0] == utc(2026, 8, 5, 5, 45)
        assert plan.free_leg_times()[-1] == utc(2026, 8, 5, 7, 0)

    def test_run_covers_horizon_with_margin(self):
        plan = self.plan()
        assert plan.horizon_seconds == 12 * 900
        assert plan.run_seconds >= plan.horizon_seconds + 3600

    def test_gfs_cycle_and_start_hour(self):
        plan = self.plan()
        assert plan.gfs_cycle == utc(2026, 8, 5, 0, 0)
        assert plan.forecast_start_hour == 4

    def test_misaligned_init_is_refused(self):
        with pytest.raises(SystemExit):
            plan_window(utc(2026, 8, 5, 5, 40), cycles=6,
                        cycle_seconds=900, free_legs=6, now=NOW)

    def test_zero_cycles_refused(self):
        with pytest.raises(SystemExit):
            self.plan(cycles=0)

    def test_payload_is_versionable(self):
        payload = self.plan().to_payload()
        assert payload["init"] == "2026-08-05T04:00:00Z"
        assert len(payload["cycle_times"]) == 6
        assert len(payload["free_leg_times"]) == 6


class TestGfsCycleSelection:
    def test_prefers_newest_available(self):
        assert latest_gfs_cycle(utc(2026, 8, 5, 4), NOW) \
            == utc(2026, 8, 5, 0)

    def test_respects_availability_lag(self):
        # at 03:30Z the 00Z cycle is 3.5 h old: not trusted yet
        assert latest_gfs_cycle(utc(2026, 8, 5, 3),
                                utc(2026, 8, 5, 3, 30)) \
            == utc(2026, 8, 4, 18)

    def test_never_after_init(self):
        # replaying an old window: newest cycle at/before init wins
        assert latest_gfs_cycle(utc(2026, 8, 1, 7), NOW) \
            == utc(2026, 8, 1, 6)


class TestLatestWindowEnd:
    def test_round_down_to_aligned_cycle(self):
        stamp = resolve_latest_window_end(
            utc(2026, 8, 5, 5, 36, 30), cycles=6, cycle_seconds=900)
        assert stamp == utc(2026, 8, 5, 5, 30)

    def test_steps_back_until_init_is_whole_hour(self):
        stamp = resolve_latest_window_end(
            utc(2026, 8, 5, 5, 36), cycles=6, cycle_seconds=1200)
        assert stamp == utc(2026, 8, 5, 5, 0)
        assert (stamp.hour * 60 + stamp.minute
                - 6 * 20) % 60 == 0


# ---------------------------------------------------------------------------
# siting
# ---------------------------------------------------------------------------
def stats(gates, east_km, north_km, valid):
    return {"gates": gates, "centroid_east_km": east_km,
            "centroid_north_km": north_km, "valid_time": valid}


class TestMotion:
    def test_pure_eastward(self):
        older = stats(5000, 0.0, 0.0, "2026-08-05T04:45:00Z")
        newer = stats(6000, 27.0, 0.0, "2026-08-05T05:30:00Z")
        motion = motion_from_centroids(older, newer, min_gates=500)
        assert motion is not None
        assert motion["u_ms"] == 10.0
        assert motion["v_ms"] == 0.0
        assert motion["toward_deg"] == 90.0

    def test_weak_echo_refuses_a_vector(self):
        older = stats(10, 0.0, 0.0, "2026-08-05T04:45:00Z")
        newer = stats(6000, 27.0, 0.0, "2026-08-05T05:30:00Z")
        assert motion_from_centroids(older, newer,
                                     min_gates=500) is None

    def test_short_baseline_refuses_a_vector(self):
        older = stats(5000, 0.0, 0.0, "2026-08-05T05:29:00Z")
        newer = stats(6000, 5.0, 0.0, "2026-08-05T05:30:00Z")
        assert motion_from_centroids(older, newer,
                                     min_gates=500) is None


class TestDomainCenter:
    def test_no_motion_centers_on_the_echo(self):
        center = site_domain_center(
            40.0, -95.0, stats(5000, -30.0, 0.0, "x"), None,
            horizon_seconds=10800, downstream_fraction=0.35,
            max_offset_km=60.0)
        assert center["basis"] == "echo centroid"
        assert center["offset_east_km"] == -30.0
        assert center["lon"] < -95.0

    def test_motion_biases_downstream(self):
        motion = {"u_ms": 10.0, "v_ms": 0.0, "speed_ms": 10.0}
        center = site_domain_center(
            40.0, -95.0, stats(5000, 0.0, 0.0, "x"), motion,
            horizon_seconds=10000, downstream_fraction=0.35,
            max_offset_km=60.0)
        assert center["offset_east_km"] == 35.0
        assert "downstream lead" in center["basis"]

    def test_offset_is_clamped(self):
        motion = {"u_ms": 30.0, "v_ms": 0.0, "speed_ms": 30.0}
        center = site_domain_center(
            40.0, -95.0, stats(5000, 100.0, 0.0, "x"), motion,
            horizon_seconds=20000, downstream_fraction=0.5,
            max_offset_km=60.0)
        assert center["offset_east_km"] == 60.0
        assert center["clamped_to_km"] == 60.0

    def test_no_echo_falls_back_to_antenna(self):
        center = site_domain_center(
            40.0, -95.0, stats(0, 0.0, 0.0, "x"), None,
            horizon_seconds=10800, downstream_fraction=0.35,
            max_offset_km=60.0)
        assert center["basis"].startswith("antenna")
        assert center["lat"] == 40.0


class TestGeojsonBox:
    def test_closed_ring_around_the_center(self):
        box = geojson_box(41.73, -93.96, 198.0)
        ring = box["coordinates"][0]
        assert len(ring) == 5
        assert ring[0] == ring[-1]
        lats = [p[1] for p in ring]
        lons = [p[0] for p in ring]
        assert min(lats) < 41.73 < max(lats)
        assert min(lons) < -93.96 < max(lons)


# ---------------------------------------------------------------------------
# stage command composition (never executed here)
# ---------------------------------------------------------------------------
class TestCommands:
    def plan(self) -> WindowPlan:
        return plan_window(utc(2026, 8, 5, 5, 30), cycles=6,
                           cycle_seconds=900, free_legs=6, now=NOW)

    def test_wrfout_name(self):
        assert wrfout_name(utc(2026, 8, 5, 4), 900) \
            == "wrfout_d01_2026-08-05_04_15_00"

    def test_wizard_cmd_carries_the_plan(self):
        argv = wizard_cmd(
            polygon=Path("box.geojson"), out_toml=Path("case.toml"),
            plan=self.plan(), profile="p", name="n", dx_km=3.0,
            source="gfs")
        text = " ".join(argv)
        assert "--cycle 2026-08-05T00" in text
        assert "--forecast-start-hour 4" in text
        assert "--root-dx 3" in text
        assert "gpuwm.cli" in text and "domain" in argv

    def test_fetch_cmd_from_wizard_hints(self):
        hints = {"cycle": "2026-08-05T00", "hours": 4,
                 "area": "1,2,3,4", "cadence": 1,
                 "forecast_start_hour": 4}
        argv = fetch_cmd(hints=hints, data_dir=Path("d"),
                         source="gfs")
        assert "--area=1,2,3,4" in argv
        assert "--cadence" in argv

    def test_obs_cmd_takes_the_site_as_argument(self):
        argv = obs_cmd(site="QQQQ", valid=utc(2026, 8, 5, 4, 15),
                       grid_wrfout=Path("g"), out_nc=Path("o"),
                       work_dir=Path("w"), bucket=None)
        assert "QQQQ" in argv
        assert "--valid-time" in argv
        assert argv[argv.index("--valid-time") + 1] \
            == "2026-08-05T04:15:00Z"

    def test_cycle_cmd_pairs_obs_with_grids(self):
        obs = [Path(f"o{k}") for k in range(6)]
        grids = [Path(f"g{k}") for k in range(6)]
        argv = cycle_cmd(
            prepared_root=Path("p"), authority_dir=Path("a"),
            profile="prof", plan=self.plan(), members=10,
            obs_files=obs, grid_wrfouts=grids, cycle_out=Path("c"),
            proof_sha="x", manifest_sha="y", content_sha="z",
            seed=1, solve_device="cuda", horizontal_loc_m=12000.0,
            vertical_loc_m=3000.0, length_scale_km=50.0,
            source="gfs")
        assert argv.count("--obs") == 6
        assert argv.count("--grid-wrfout") == 6
        assert "--save-composites" in argv
        assert "--free-legs" in argv

    def test_stop_after_orders_stages(self):
        assert STAGES.index("survey") < STAGES.index("obs") \
            < STAGES.index("cycle") < STAGES.index("render")
        assert _stop("survey", "survey")
        assert _stop("obs", "cycle")       # later stage stops too
        assert not _stop("cycle", "obs")   # earlier stage continues
        assert _stop(None, "render") is False


# ---------------------------------------------------------------------------
# the CLI surface itself
# ---------------------------------------------------------------------------
class TestParser:
    def test_run_defaults_are_mechanism_numbers(self):
        args = build_parser().parse_args(
            ["run", "--site", "qqqq", "--window-end", "latest",
             "--out", "case"])
        assert args.site == "QQQQ"
        assert args.cycles == 6
        assert args.cycle_seconds == 900
        assert args.free_legs == 6
        assert args.members == 10
        assert args.solve_device == "cuda"
        assert args.stop_after is None

    def test_verify_requires_case_dir(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["verify"])

    def test_mode_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_verification_is_on_by_default(self):
        args = build_parser().parse_args(
            ["run", "--site", "qqqq", "--window-end", "latest",
             "--out", "case"])
        assert args.no_verify is False

    def test_no_verify_opts_out(self):
        args = build_parser().parse_args(
            ["run", "--site", "qqqq", "--window-end", "latest",
             "--out", "case", "--no-verify"])
        assert args.no_verify is True

    def test_watch_mode_exists_for_an_existing_case(self):
        args = build_parser().parse_args(
            ["watch", "--case-dir", "case", "--poll-seconds", "30"])
        assert args.mode == "watch"
        assert args.case_dir == Path("case")
        assert args.poll_seconds == 30
        assert args.max_minutes > 0


# ---------------------------------------------------------------------------
# the verification state machine: what a GUI polls
# ---------------------------------------------------------------------------
FREE_TIMES = ["2026-08-05T05:45:00Z", "2026-08-05T06:00:00Z",
              "2026-08-05T06:15:00Z"]


def row(valid: str, fss: float = 0.75) -> dict:
    """A per-frame row shaped like the renderer's published ones."""

    return {"leg": 6, "valid": valid,
            "obs_valid_time": valid.replace(":00Z", ":12Z"),
            "obs_cols_gt35": 2817, "fcst_cols_gt35_in_echo": 1899,
            "control_cols_gt35_in_echo": 519, "fss30_fcst": fss,
            "fss30_control": 0.24, "fss_half_width_cells": 4}


class TestFrames:
    def test_initial_frames_are_pending_in_order(self):
        frames = initial_frames(FREE_TIMES)
        assert [f["valid"] for f in frames] == FREE_TIMES
        assert {f["status"] for f in frames} == {"pending"}

    def test_obs_name_carries_the_site_argument(self):
        name = verify_obs_name("QQQQ", utc(2026, 8, 5, 6, 15))
        assert name == "verify-qqqq-202608050615.nc"

    def test_merge_grades_only_frames_with_numbers(self):
        frames = initial_frames(FREE_TIMES)
        merged = merge_gallery_rows(frames, [row(FREE_TIMES[0])])
        assert merged[0]["status"] == "verified"
        assert merged[0]["fss30_fcst"] == 0.75
        assert [f["status"] for f in merged[1:]] == ["pending"] * 2

    def test_merge_clears_a_stale_note_and_keeps_valid_time(self):
        frames = [{"valid": FREE_TIMES[0], "status": "pending",
                   "note": "not covered yet"}]
        merged = merge_gallery_rows(frames, [row(FREE_TIMES[0])])
        assert "note" not in merged[0]
        assert merged[0]["valid"] == FREE_TIMES[0]

    def test_merge_leaves_the_input_untouched(self):
        frames = initial_frames(FREE_TIMES)
        merge_gallery_rows(frames, [row(FREE_TIMES[0])])
        assert frames[0]["status"] == "pending"

    def test_a_built_observation_alone_is_not_a_grade(self):
        frames = [{"valid": FREE_TIMES[0], "status": "pending",
                   "obs_file": "verify-qqqq-202608050545.nc"}]
        assert merge_gallery_rows(frames, [])[0]["status"] == "pending"


class TestStateMachine:
    def test_rolling_while_frames_wait(self):
        frames = merge_gallery_rows(initial_frames(FREE_TIMES),
                                    [row(FREE_TIMES[0])])
        assert advance_state(frames, exhausted=False) == "rolling"

    def test_complete_when_every_frame_carries_numbers(self):
        frames = merge_gallery_rows(initial_frames(FREE_TIMES),
                                    [row(t) for t in FREE_TIMES])
        assert advance_state(frames, exhausted=False) == "complete"

    def test_incomplete_when_the_verifier_gives_up(self):
        frames = initial_frames(FREE_TIMES)
        assert advance_state(frames, exhausted=True) == "incomplete"

    def test_no_free_legs_is_vacuously_complete(self):
        assert advance_state([], exhausted=False) == "complete"

    def test_handoff_defaults_to_pending_not_off(self):
        frames = initial_frames(FREE_TIMES)
        assert handoff_state(no_verify=False, frames=frames) \
            == "pending"

    def test_handoff_respects_the_opt_out(self):
        assert handoff_state(no_verify=True,
                             frames=initial_frames(FREE_TIMES)) \
            == "disabled"

    def test_handoff_with_nothing_to_grade(self):
        assert handoff_state(no_verify=False, frames=[]) == "complete"

    def test_verdicts_count_what_is_graded(self):
        frames = merge_gallery_rows(initial_frames(FREE_TIMES),
                                    [row(FREE_TIMES[0])])
        assert "1/3" in verdict_line(frames, "rolling")
        assert "not started" in verdict_line(frames, "disabled")

    def test_block_is_versioned_and_counted(self):
        frames = merge_gallery_rows(initial_frames(FREE_TIMES),
                                    [row(FREE_TIMES[0])])
        block = verification_block(frames=frames, state="rolling")
        assert block["schema"] == VERIFY_SCHEMA
        assert (block["graded"], block["total"]) == (1, 3)
        assert block["frames"] is frames
        assert "unscored" in block["honesty"]

    def test_block_refuses_an_unknown_state(self):
        with pytest.raises(SystemExit):
            verification_block(frames=[], state="probably-fine")


# ---------------------------------------------------------------------------
# the auto-handoff: a finished run starts its own verifier
# ---------------------------------------------------------------------------
def seed_case(tmp_path: Path, free_times=FREE_TIMES) -> Path:
    """A case directory with just the receipt the verifier reads."""

    (tmp_path / "nowcast-receipt.json").write_text(json.dumps({
        "schema": "gpuwm-da.nowcast.v1", "site": "QQQQ",
        "plan": {"init": "2026-08-05T04:00:00Z",
                 "cycle_seconds": 900,
                 "free_leg_times": list(free_times)}}), encoding="utf-8")
    return tmp_path


class TestWatchCommand:
    def test_watch_cmd_runs_this_module(self):
        argv = watch_cmd(case_dir=Path("case"), poll_seconds=240,
                         max_minutes=180, max_offset_seconds=480.0,
                         bucket=None)
        assert argv[1:4] == ["-m", "tools.da_nowcast", "watch"]
        assert argv[argv.index("--case-dir") + 1] == "case"
        assert "--bucket" not in argv

    def test_watch_cmd_forwards_the_bucket_override(self):
        argv = watch_cmd(case_dir=Path("case"), poll_seconds=30,
                         max_minutes=5, max_offset_seconds=60.0,
                         bucket="some-bucket")
        assert argv[argv.index("--bucket") + 1] == "some-bucket"
        assert argv[argv.index("--poll-seconds") + 1] == "30"

    def test_handoff_spawns_detached_and_records_the_state(
            self, tmp_path, monkeypatch):
        case = seed_case(tmp_path)
        seen = {}

        def fake_spawn(argv, *, cwd, log_path):
            seen["argv"] = argv
            seen["log"] = log_path
            return 4242

        monkeypatch.setattr("tools.da_nowcast.spawn_detached",
                            fake_spawn)
        block = start_verification(
            case, frames=initial_frames(FREE_TIMES), poll_seconds=240,
            max_minutes=180, max_offset_seconds=480.0, bucket=None,
            repo_root=tmp_path)

        assert "watch" in seen["argv"]
        assert seen["log"] == case / "verify-watch.log"
        assert block["state"] == "rolling"
        assert block["watcher"]["pid"] == 4242
        assert block["watcher"]["argv"] == seen["argv"]

        receipt = json.loads(
            (case / "nowcast-receipt.json").read_text(encoding="utf-8"))
        assert receipt["verification"]["state"] == "rolling"
        assert receipt["verification"]["total"] == 3
        assert receipt["site"] == "QQQQ"   # the rest of it survives

    def test_receipt_rewrite_is_atomic_and_key_scoped(self, tmp_path):
        case = seed_case(tmp_path)
        write_verification(case, verification_block(
            frames=initial_frames(FREE_TIMES), state="pending"))
        write_verification(case, verification_block(
            frames=initial_frames(FREE_TIMES), state="rolling"))
        receipt = json.loads(
            (case / "nowcast-receipt.json").read_text(encoding="utf-8"))
        assert receipt["verification"]["state"] == "rolling"
        assert receipt["plan"]["cycle_seconds"] == 900
        assert not list(case.glob("*.tmp"))


class TestFooterBand:
    """The credit line gets reserved geometry, not luck."""

    def test_short_wide_figures_reserve_a_visible_band(self):
        # 02-cycle-numbers is 5.2in tall; its tick labels used to run
        # straight into the footer.
        assert footer_band_fraction(5.2) > 0.05

    def test_tall_figures_reserve_proportionally_less(self):
        assert footer_band_fraction(10.4) < footer_band_fraction(5.2)

    def test_the_band_never_eats_the_figure(self):
        assert footer_band_fraction(0.5) <= 0.12

    def test_a_zero_height_figure_is_refused(self):
        with pytest.raises(ValueError):
            footer_band_fraction(0.0)


class TestVerifyPass:
    def test_gallery_rows_absent_is_not_an_error(self, tmp_path):
        assert gallery_rows(tmp_path) == []

    def test_a_pass_grades_from_the_renderers_published_rows(
            self, tmp_path, monkeypatch):
        case = seed_case(tmp_path)
        (case / "obsverify").mkdir()
        (case / "obsverify" / verify_obs_name(
            "QQQQ", datetime(2026, 8, 5, 5, 45,
                             tzinfo=timezone.utc))).write_bytes(b"x")
        (case / "gallery").mkdir()
        (case / "gallery" / "_verification.json").write_text(
            json.dumps({"frames": [row(FREE_TIMES[0])]}),
            encoding="utf-8")

        asked = []

        def record(*, valid, **kwargs):
            asked.append(valid.strftime("%Y-%m-%dT%H:%M:00Z"))
            return False, "archive does not cover it yet"

        monkeypatch.setattr("tools.da_nowcast.build_verify_frame",
                            record)
        frames, built = verify_pass(
            case_dir=case, site="QQQQ",
            init=utc(2026, 8, 5, 4), frames=initial_frames(FREE_TIMES),
            bucket=None, max_offset_seconds=480.0, repo_root=case)
        assert built == 0
        assert FREE_TIMES[0] not in asked   # already on disk, not refetched
        assert asked == FREE_TIMES[1:]
        assert frames[0]["status"] == "verified"
        assert frames[0]["obs_file"].endswith("202608050545.nc")
        assert frames[1]["note"] == "archive does not cover it yet"

    def test_a_pass_never_asks_for_a_future_valid_time(
            self, tmp_path, monkeypatch):
        soon = datetime.now(timezone.utc) + timedelta(hours=2)
        case = seed_case(tmp_path,
                         [soon.strftime("%Y-%m-%dT%H:%M:00Z")])

        def refuse(**kwargs):
            raise AssertionError("asked the archive for the future")

        monkeypatch.setattr("tools.da_nowcast.build_verify_frame",
                            refuse)
        frames, built = verify_pass(
            case_dir=case, site="QQQQ", init=utc(2026, 8, 5, 4),
            frames=initial_frames([soon.strftime("%Y-%m-%dT%H:%M:00Z")]),
            bucket=None, max_offset_seconds=480.0, repo_root=case)
        assert built == 0
        assert frames[0]["status"] == "pending"
        assert "not reached" in frames[0]["note"]


# ---------------------------------------------------------------------------
# the gallery dates its frames from the report, not from a cadence
# ---------------------------------------------------------------------------
class TestGalleryLegTimes:
    """A nowcast that cycles on the radar's rhythm has legs of unequal
    length, so every caption has to come from the leg's own end."""

    def gallery(self, ends, *, cycles=2, notice=None):
        from tools.da_nowcast_render import Gallery

        g = Gallery.__new__(Gallery)
        g.init = utc(2026, 8, 5, 4, 0)
        g.leg_seconds = 900.0
        g.leg_end_s = list(ends)
        g.legs = [{"end_s": e} for e in ends]
        g.cycles = cycles
        g.notice = notice
        return g

    def test_valid_times_come_from_the_report(self):
        g = self.gallery([1260.0, 2505.0, 3405.0])
        assert g.leg_valid(0) == utc(2026, 8, 5, 4, 21)
        assert g.leg_valid(1) == utc(2026, 8, 5, 4, 41, 45)

    def test_a_report_without_leg_ends_falls_back_to_the_cadence(self):
        from tools.da_nowcast_render import Gallery

        g = Gallery.__new__(Gallery)
        g.init = utc(2026, 8, 5, 4, 0)
        g.leg_seconds = 900.0
        g.leg_end_s = []
        assert g.leg_elapsed(0) == 900.0
        assert g.leg_elapsed(3) == 3600.0

    def test_an_observed_frame_matches_the_leg_it_belongs_to(self):
        g = self.gallery([1260.0, 2505.0, 3405.0])
        assert g.leg_at(2505.0) == 1
        assert g.leg_at(2500.0) == 1

    def test_a_frame_between_legs_matches_nothing(self):
        g = self.gallery([600.0, 1200.0, 1800.0])
        assert g.leg_at(900.0) is None

    def test_equal_legs_report_one_cadence(self):
        assert self.gallery([900.0, 1800.0, 2700.0],
                            cycles=3).cadence_label() == "15-min"

    def test_unequal_legs_report_the_range_they_used(self):
        assert self.gallery([300.0, 660.0, 960.0],
                            cycles=3).cadence_label() == "5-6-min"

    def test_no_notice_means_no_reserved_band_and_no_forced_title(self):
        g = self.gallery([900.0])
        assert g.suptitle_y(_FakeFig(6.0)) is None

    def test_a_notice_pushes_the_title_below_its_band(self):
        g = self.gallery([900.0], notice={"level": "warn",
                                          "headline": "late"})
        assert 0.9 < g.suptitle_y(_FakeFig(6.0)) < 1.0


class _FakeFig:
    def __init__(self, height):
        self._h = height

    def get_figheight(self):
        return self._h


# ---------------------------------------------------------------------------
# the georeference cadence the run needs, not the one a forecast wants
# ---------------------------------------------------------------------------
class TestHistoryCadence:
    TOML = (
        "[experiment]\nname = 'x'\n\n"
        "[[domain]]\ngrid_id = 1\nnx = 132\nny = 132\n"
        "history_interval_s = 3600.0\nradt = 1.0\n")

    def test_the_root_cadence_is_rewritten(self):
        from tools.da_nowcast import retime_history

        text, was = retime_history(self.TOML, 300.0)
        assert was == 3600.0
        assert "history_interval_s = 300" in text

    def test_only_that_key_changes(self):
        from tools.da_nowcast import retime_history

        text, _ = retime_history(self.TOML, 300.0)
        before = [ln for ln in self.TOML.splitlines()
                  if "history_interval_s" not in ln]
        after = [ln for ln in text.splitlines()
                 if "history_interval_s" not in ln]
        assert before == after

    def test_only_the_first_domain_is_touched(self):
        from tools.da_nowcast import retime_history

        two = self.TOML + "\n[[domain]]\nhistory_interval_s = 900.0\n"
        text, _ = retime_history(two, 300.0)
        assert text.count("history_interval_s = 900") == 1
        assert text.count("history_interval_s = 300") == 1

    def test_a_wizard_that_stopped_emitting_it_is_a_refusal(self):
        from tools.da_nowcast import FrontDoorError, retime_history

        with pytest.raises(FrontDoorError, match="no history_interval_s"):
            retime_history("[experiment]\nname = 'x'\n", 300.0)

    def test_the_flag_exists_and_defaults_to_the_cycle_cadence(self):
        args = build_parser().parse_args(
            ["run", "--site", "qqqq", "--window-end", "latest",
             "--out", "o"])
        assert args.history_interval_seconds is None
        assert args.cycle_seconds == 900


# ---------------------------------------------------------------------------
# the card a plan was sized against is the card the run is sized against
# ---------------------------------------------------------------------------
class TestCardCarriesThrough:
    def plan(self):
        # window end at hh:05 with one 300 s cycle puts init on the
        # hour, which is the front door's rule
        return plan_window(utc(2026, 8, 5, 6, 5), cycles=1,
                           cycle_seconds=300, free_legs=0, now=NOW,
                           run_hours=4)

    def test_no_card_named_leaves_the_wizard_its_default(self):
        argv = wizard_cmd(polygon=Path("b"), out_toml=Path("t"),
                          plan=self.plan(), profile="p", name="n",
                          dx_km=3.0, source="gfs")
        assert "--vram-gib" not in argv

    def test_a_named_card_reaches_the_wizard(self):
        argv = wizard_cmd(polygon=Path("b"), out_toml=Path("t"),
                          plan=self.plan(), profile="p", name="n",
                          dx_km=3.0, source="gfs", vram_gib=31.84)
        assert argv[argv.index("--vram-gib") + 1] == "31.84"

    def test_the_front_door_takes_one(self):
        args = build_parser().parse_args(
            ["run", "--site", "qqqq", "--window-end", "latest",
             "--out", "o", "--vram-gib", "32"])
        assert args.vram_gib == 32.0


# ---------------------------------------------------------------------------
# a gallery that redraws itself cannot be checked by eye each time
# ---------------------------------------------------------------------------
class TestCreditAndTitleFit:
    FOOT = ("gpuwm radar-DA nowcast demo (N=6) — real QQQQ NEXRAD "
            "Level-II — UNSCORED demo, not campaign evidence — basemap: "
            "Natural Earth 10m + US Census counties (vendored assets)")
    SRC = "source: ArWen (model) · QQQQ Level-II (obs)"

    def test_a_wide_figure_keeps_the_two_sided_credit(self):
        from tools.da_nowcast_render import credit_layout

        assert credit_layout(24.0, self.FOOT, self.SRC)["mode"] == "sides"

    def test_a_narrow_figure_stacks_instead_of_overprinting(self):
        from tools.da_nowcast_render import credit_layout

        assert credit_layout(6.0, self.FOOT, self.SRC)["mode"] \
            == "stacked"

    def test_a_side_by_side_credit_actually_fits(self):
        from tools.da_nowcast_render import credit_layout, text_width_in

        for width in (9.6, 12.0, 14.4, 19.2, 24.0):
            layout = credit_layout(width, self.FOOT, self.SRC)
            if layout["mode"] != "sides":
                continue
            used = (text_width_in(self.FOOT, layout["foot_pt"])
                    + text_width_in(self.SRC, layout["src_pt"]))
            assert used <= width, width

    def test_a_title_that_fits_is_left_alone(self):
        from tools.da_nowcast_render import fit_title

        text, points = fit_title("short title", 20.0)
        assert (text, points) == ("short title", 12.0)

    def test_a_long_title_shrinks_before_it_wraps(self):
        from tools.da_nowcast_render import fit_title, text_width_in

        long = "x" * 160
        text, points = fit_title(long, 14.0)
        assert "\n" not in text
        assert points < 12.0
        assert text_width_in(text, points) <= 14.0

    def test_a_title_that_cannot_shrink_enough_wraps(self):
        from tools.da_nowcast_render import fit_title, text_width_in

        text, points = fit_title(" ".join(["word"] * 60), 8.0)
        assert "\n" in text
        assert points >= 8.5
        for line in text.splitlines():
            assert text_width_in(line, points) <= 8.0


# ---------------------------------------------------------------------------
# a length scale the domain can actually carry
# ---------------------------------------------------------------------------
class TestPerturbationScale:
    """A caller who draws a smaller box has not asked for different
    science; they have asked for a smaller box."""

    def test_the_bounds_come_from_the_perturbation_lane_itself(self):
        from tools.da_nowcast import perturbation_scale_bounds

        # pins the import: a rename in gpuwm.da.perturb fails here
        # rather than three minutes into somebody's run
        floor, ceiling = perturbation_scale_bounds(
            nx=132, ny=132, dx_m=3000.0, dy_m=3000.0)
        assert floor == pytest.approx(6.0)
        assert ceiling == pytest.approx(396.0 / (2 * 3.141592653589793))

    def test_a_big_box_keeps_the_default(self):
        from tools.da_nowcast import (DEFAULT_LENGTH_SCALE_KM,
                                      resolvable_length_scale_km)

        scale, note = resolvable_length_scale_km(
            nx=132, ny=132, dx_m=3000.0, dy_m=3000.0)
        assert scale == DEFAULT_LENGTH_SCALE_KM
        assert note is None

    def test_a_small_box_is_capped_and_says_so(self):
        from tools.da_nowcast import resolvable_length_scale_km

        scale, note = resolvable_length_scale_km(
            nx=86, ny=94, dx_m=3000.0, dy_m=3000.0)
        assert scale < 50.0
        assert "span/(2*pi)" in note

    def test_the_capped_value_is_one_the_lane_accepts(self):
        from gpuwm.da.perturb import (FieldPerturbation, _check_resolvable)
        from tools.da_nowcast import resolvable_length_scale_km

        for nx, ny, dx in ((86, 94, 3000.0), (132, 132, 3000.0),
                           (60, 60, 3000.0), (400, 300, 12000.0)):
            scale, _ = resolvable_length_scale_km(
                nx=nx, ny=ny, dx_m=dx, dy_m=dx)
            spec = FieldPerturbation.from_mapping({
                "name": "u", "amplitude": 1.5,
                "length_scale_km": scale})
            # the artifact, not a model of it: the lane's own checker
            _check_resolvable((49, ny, nx), dx / 1000.0, dx / 1000.0,
                              spec)

    def test_an_explicit_request_inside_the_range_is_honoured(self):
        from tools.da_nowcast import resolvable_length_scale_km

        scale, note = resolvable_length_scale_km(
            nx=86, ny=94, dx_m=3000.0, dy_m=3000.0, requested=20.0)
        assert (scale, note) == (20.0, None)

    def test_an_explicit_request_outside_it_is_capped_and_named(self):
        from tools.da_nowcast import resolvable_length_scale_km

        scale, note = resolvable_length_scale_km(
            nx=86, ny=94, dx_m=3000.0, dy_m=3000.0, requested=500.0)
        assert scale < 500.0
        assert "--length-scale-km 500" in note

    def test_a_domain_with_no_usable_scale_is_a_refusal(self):
        from tools.da_nowcast import FrontDoorError, resolvable_length_scale_km

        with pytest.raises(FrontDoorError, match="too small to perturb"):
            resolvable_length_scale_km(nx=8, ny=8, dx_m=3000.0,
                                       dy_m=3000.0)

    def test_the_flag_defaults_to_derived(self):
        args = build_parser().parse_args(
            ["run", "--site", "qqqq", "--window-end", "latest",
             "--out", "o"])
        assert args.length_scale_km is None
