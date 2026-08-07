"""The continuous nowcast daemon's planning, refusals and bookkeeping.

Pure tests: no network, no GPU, no subprocess execution.  Sites here are
synthetic ids -- real station names never enter the tree's generic code
or its fixtures (standing owner rule).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.da_nowcast_auto import (
    EXPECTED_VOLUME_INTERVAL_SECONDS, NOTICE_SCHEMA, SCHEMA, AutoError,
    assemble_view_report, bootstrap_cmd, build_parser, epoch_has_room,
    fingerprint_changed, free_forecast_seconds, loop_argv,
    nearest_grid_wrfout, notice_payload, pick_next_volume, plan_leg,
    read_status, should_render, snap_to_step, status_path, stop_path,
    usable_volumes, verdict_line, volume_is_late, volumes_behind,
    wrfout_time, write_status)


def utc(*parts) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


INIT = utc(2026, 8, 5, 4, 0)


def volume(minute: int, *, hour: int = 4, name: str | None = None
           ) -> dict:
    stamp = utc(2026, 8, 5, hour, minute)
    return {"filename": name or f"VOL{hour:02d}{minute:02d}",
            "key": f"k/{hour:02d}{minute:02d}",
            "valid_time": stamp.strftime("%Y-%m-%dT%H:%M:%SZ")}


# ---------------------------------------------------------------------------
# snapping to the model's own step lattice
# ---------------------------------------------------------------------------
class TestSnap:
    def test_rounds_to_the_nearest_step(self):
        assert snap_to_step(907.0, 15.0) == 900.0
        assert snap_to_step(913.0, 15.0) == 915.0

    def test_a_step_boundary_is_left_alone(self):
        assert snap_to_step(900.0, 15.0) == 900.0

    def test_zero_step_refused(self):
        with pytest.raises(AutoError, match="positive"):
            snap_to_step(60.0, 0.0)


# ---------------------------------------------------------------------------
# one leg, on the radar's cadence
# ---------------------------------------------------------------------------
class TestPlanLeg:
    def test_advances_to_the_volume_time(self):
        plan = plan_leg(init=INIT, elapsed_s=0.0,
                        volume_time=utc(2026, 8, 5, 4, 5),
                        dt_s=15.0)
        assert plan.leg_seconds == 300.0
        assert plan.end_elapsed_s == 300.0
        assert plan.valid == utc(2026, 8, 5, 4, 5)
        assert plan.snap_offset_s == 0.0

    def test_unequal_cadence_is_normal_not_an_error(self):
        first = plan_leg(init=INIT, elapsed_s=0.0,
                         volume_time=utc(2026, 8, 5, 4, 5), dt_s=15.0)
        second = plan_leg(init=INIT, elapsed_s=first.end_elapsed_s,
                          volume_time=utc(2026, 8, 5, 4, 11),
                          dt_s=15.0)
        assert (first.leg_seconds, second.leg_seconds) == (300.0, 360.0)

    def test_off_lattice_volume_is_snapped_and_the_offset_kept(self):
        plan = plan_leg(init=INIT, elapsed_s=0.0,
                        volume_time=utc(2026, 8, 5, 4, 5, 7),
                        dt_s=15.0)
        assert plan.end_elapsed_s == 300.0
        assert plan.snap_offset_s == pytest.approx(-7.0)

    def test_a_volume_already_passed_is_refused(self):
        with pytest.raises(AutoError, match="nothing to advance"):
            plan_leg(init=INIT, elapsed_s=600.0,
                     volume_time=utc(2026, 8, 5, 4, 10), dt_s=15.0)

    def test_a_sliver_leg_is_refused(self):
        with pytest.raises(AutoError, match="sliver"):
            plan_leg(init=INIT, elapsed_s=300.0,
                     volume_time=utc(2026, 8, 5, 4, 6), dt_s=15.0)

    def test_a_feed_gap_is_refused_as_a_gap(self):
        with pytest.raises(AutoError, match="feed gap"):
            plan_leg(init=INIT, elapsed_s=0.0,
                     volume_time=utc(2026, 8, 5, 5, 0), dt_s=15.0)

    def test_payload_is_json_ready(self):
        payload = plan_leg(init=INIT, elapsed_s=0.0,
                           volume_time=utc(2026, 8, 5, 4, 5),
                           dt_s=15.0).to_payload()
        json.dumps(payload)
        assert payload["valid"] == "2026-08-05T04:05:00Z"


# ---------------------------------------------------------------------------
# which volume comes next
# ---------------------------------------------------------------------------
class TestVolumeSelection:
    def test_metadata_companions_are_not_volumes(self):
        listing = {"volumes": [volume(5), {**volume(6),
                                           "filename": "X_MDM"}]}
        assert [v["filename"] for v in usable_volumes(listing)] \
            == ["VOL0405"]

    def test_listing_is_sorted_oldest_first(self):
        listing = {"volumes": [volume(11), volume(5), volume(17)]}
        assert [v["valid_time"] for v in usable_volumes(listing)] == [
            "2026-08-05T04:05:00Z", "2026-08-05T04:11:00Z",
            "2026-08-05T04:17:00Z"]

    def test_oldest_unused_volume_is_taken_so_catch_up_uses_real_data(
            self):
        vols = [volume(5), volume(11), volume(17)]
        picked = pick_next_volume(vols, after=INIT, min_gap_s=120.0)
        assert picked["filename"] == "VOL0405"

    def test_a_volume_inside_the_floor_is_skipped(self):
        vols = [volume(1), volume(11)]
        picked = pick_next_volume(vols, after=INIT, min_gap_s=120.0)
        assert picked["filename"] == "VOL0411"

    def test_none_when_nothing_is_new_enough(self):
        assert pick_next_volume([volume(5)],
                                after=utc(2026, 8, 5, 4, 5),
                                min_gap_s=120.0) is None

    def test_behind_counts_only_what_is_ahead_of_the_analysis(self):
        vols = [volume(5), volume(11), volume(17)]
        assert volumes_behind(vols, after=utc(2026, 8, 5, 4, 11)) == 1
        assert volumes_behind(vols, after=INIT) == 3


# ---------------------------------------------------------------------------
# the epoch's boundary-data horizon
# ---------------------------------------------------------------------------
class TestEpoch:
    def test_room_for_another_cycle(self):
        assert epoch_has_room(elapsed_s=3600.0, run_seconds=14400.0,
                              free_forecast_s=5400.0, margin_s=1800.0)

    def test_no_room_is_the_end_of_the_epoch(self):
        assert not epoch_has_room(elapsed_s=8000.0, run_seconds=14400.0,
                                  free_forecast_s=5400.0,
                                  margin_s=1800.0)

    def test_exactly_full_still_counts_as_room(self):
        assert epoch_has_room(elapsed_s=7200.0, run_seconds=14400.0,
                              free_forecast_s=5400.0, margin_s=1800.0)

    def test_free_forecast_length_is_legs_times_leg(self):
        assert free_forecast_seconds(6, 900.0) == 5400.0
        assert free_forecast_seconds(0, 900.0) == 0.0


# ---------------------------------------------------------------------------
# the georeference an observation is gridded onto
# ---------------------------------------------------------------------------
class TestGridChoice:
    def test_reads_the_time_out_of_the_name(self):
        assert wrfout_time("wrfout_d01_2026-08-05_04_15_00") == \
            utc(2026, 8, 5, 4, 15)

    def test_a_name_that_is_not_one_returns_none(self):
        assert wrfout_time("cycle-report.json") is None
        assert wrfout_time("wrfout_d01_not-a-time") is None

    def test_nearest_in_time_wins(self):
        paths = [Path("wrfout_d01_2026-08-05_04_00_00"),
                 Path("wrfout_d01_2026-08-05_04_15_00"),
                 Path("wrfout_d01_2026-08-05_04_30_00")]
        chosen, offset = nearest_grid_wrfout(
            paths, utc(2026, 8, 5, 4, 17))
        assert chosen.name.endswith("04_15_00")
        assert offset == 120.0

    def test_nothing_to_grid_onto_is_a_refusal(self):
        with pytest.raises(AutoError, match="no georeference"):
            nearest_grid_wrfout([], utc(2026, 8, 5, 4, 17))

    def test_too_far_in_time_is_a_refusal(self):
        paths = [Path("wrfout_d01_2026-08-05_04_00_00")]
        with pytest.raises(AutoError, match="ceiling"):
            nearest_grid_wrfout(paths, utc(2026, 8, 5, 6, 0))


# ---------------------------------------------------------------------------
# the view the gallery renders
# ---------------------------------------------------------------------------
def cycle_report(*, observed_end: float, free_ends: list[float],
                 members: int = 10) -> dict:
    legs = [{"leg": 0, "start_s": observed_end - 300.0,
             "end_s": observed_end, "analysis": {"applied": True},
             "trajectories": {}}]
    for index, end in enumerate(free_ends):
        legs.append({"leg": index + 1, "start_s": end - 900.0,
                     "end_s": end, "trajectories": {}})
    return {"schema": "gpuwm-da.prepared-cycle-report.v1",
            "args": {"members": str(members), "leg_seconds": "300.0",
                     "free_legs": str(len(free_ends)),
                     "solve_device": "cuda"},
            "legs": legs}


class TestViewAssembly:
    def test_one_observed_leg_per_cycle_then_the_newest_free_legs(self):
        reports = [cycle_report(observed_end=300.0, free_ends=[1200.0]),
                   cycle_report(observed_end=600.0,
                                free_ends=[1500.0, 2400.0])]
        view = assemble_view_report(reports, free_legs=2)
        assert [leg["end_s"] for leg in view["legs"]] == [
            300.0, 600.0, 1500.0, 2400.0]
        assert view["args"]["free_legs"] == 2
        assert view["cycles"] == 2

    def test_superseded_free_legs_are_dropped(self):
        reports = [cycle_report(observed_end=300.0,
                                free_ends=[1200.0, 2100.0]),
                   cycle_report(observed_end=600.0, free_ends=[1500.0])]
        view = assemble_view_report(reports, free_legs=1)
        assert 1200.0 not in [leg["end_s"] for leg in view["legs"]]

    def test_a_catch_up_view_has_no_free_legs_and_says_so(self):
        reports = [cycle_report(observed_end=300.0, free_ends=[]),
                   cycle_report(observed_end=600.0, free_ends=[])]
        view = assemble_view_report(reports, free_legs=0)
        assert view["args"]["free_legs"] == 0
        assert len(view["legs"]) == 2

    def test_no_cycles_is_a_refusal_not_an_empty_view(self):
        with pytest.raises(AutoError, match="no cycles"):
            assemble_view_report([], free_legs=6)

    def test_a_cycle_without_an_observed_leg_is_a_refusal(self):
        broken = cycle_report(observed_end=300.0, free_ends=[])
        broken["legs"][0].pop("analysis")
        with pytest.raises(AutoError, match="no observed leg"):
            assemble_view_report([broken], free_legs=0)

    def test_members_survive_into_the_view_the_renderer_reads(self):
        view = assemble_view_report(
            [cycle_report(observed_end=300.0, free_ends=[], members=36)],
            free_legs=0)
        assert view["args"]["members"] == "36"


# ---------------------------------------------------------------------------
# saying it out loud
# ---------------------------------------------------------------------------
class TestNotices:
    def test_payload_carries_its_schema_and_level(self):
        payload = notice_payload(level="warn", headline="h", detail="d")
        assert payload["schema"] == NOTICE_SCHEMA
        assert payload["level"] == "warn"

    def test_an_unknown_level_is_refused(self):
        with pytest.raises(AutoError, match="unknown notice level"):
            notice_payload(level="disaster", headline="h", detail="d")

    def test_a_quiet_feed_becomes_late_after_two_intervals(self):
        last = utc(2026, 8, 5, 4, 0)
        soon = last + timedelta(
            seconds=EXPECTED_VOLUME_INTERVAL_SECONDS + 60)
        later = last + timedelta(
            seconds=EXPECTED_VOLUME_INTERVAL_SECONDS + 900)
        assert not volume_is_late(last_valid=last, now=soon)
        assert volume_is_late(last_valid=last, now=later)

    def test_no_previous_volume_is_not_lateness(self):
        assert not volume_is_late(last_valid=None,
                                  now=utc(2026, 8, 5, 9, 0))

    def test_verdict_names_the_mode(self):
        assert "catching up" in verdict_line(
            "catching-up", cycles=3, site="QQQQ", behind=4)
        assert "waiting" in verdict_line("waiting", cycles=3,
                                         site="QQQQ")
        assert "stopped" in verdict_line("stopped", cycles=3,
                                         site="QQQQ")

    def test_a_warning_notice_becomes_the_verdict(self):
        notice = notice_payload(level="warn", headline="feed is late",
                                detail="d")
        assert verdict_line("waiting", cycles=3, site="QQQQ",
                            notice=notice) == "QQQQ: feed is late"


# ---------------------------------------------------------------------------
# when to spend seconds on a redraw
# ---------------------------------------------------------------------------
class TestRenderCadence:
    def test_a_current_analysis_always_redraws(self):
        assert should_render(cycle_index=0, caught_up=True,
                             render_every=4)

    def test_catch_up_redraws_on_the_interval(self):
        drawn = [i for i in range(8)
                 if should_render(cycle_index=i, caught_up=False,
                                  render_every=4)]
        assert drawn == [3, 7]

    def test_render_every_zero_means_never_during_catch_up(self):
        assert not should_render(cycle_index=9, caught_up=False,
                                 render_every=0)


# ---------------------------------------------------------------------------
# the run root must not move underneath a run
# ---------------------------------------------------------------------------
class TestWorktreeFingerprint:
    def test_unchanged_is_none(self):
        fp = {"head": "a" * 40, "dirty_paths": 3, "is_git": True}
        assert fingerprint_changed(fp, dict(fp)) is None

    def test_a_commit_is_caught(self):
        before = {"head": "a" * 40, "dirty_paths": 3, "is_git": True}
        after = {"head": "b" * 40, "dirty_paths": 0, "is_git": True}
        assert "HEAD moved" in fingerprint_changed(before, after)

    def test_a_staged_edit_is_caught(self):
        before = {"head": "a" * 40, "dirty_paths": 3, "is_git": True}
        after = {"head": "a" * 40, "dirty_paths": 5, "is_git": True}
        assert "uncommitted-file count" in fingerprint_changed(before,
                                                               after)

    def test_a_non_git_tree_is_not_policed(self):
        before = {"head": "", "dirty_paths": 0, "is_git": False}
        after = {"head": "", "dirty_paths": 9, "is_git": False}
        assert fingerprint_changed(before, after) is None


# ---------------------------------------------------------------------------
# the argument surface and the commands it builds
# ---------------------------------------------------------------------------
class TestArguments:
    def parsed(self, *extra):
        return build_parser().parse_args(
            ["start", "--site", "qqqq", "--out", "o", *extra])

    def test_site_is_validated_and_uppercased(self):
        assert self.parsed().site == "QQQQ"

    def test_a_bad_site_is_refused(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["start", "--site", "1234", "--out", "o"])

    def test_ensemble_size_is_an_argument_with_a_stated_default(self):
        assert self.parsed().members == 10
        assert self.parsed("--members", "36").members == 36

    def test_it_runs_forever_by_default(self):
        assert self.parsed().max_cycles == 0
        assert self.parsed().max_epochs == 0

    def test_bootstrap_stops_at_the_forecast_and_asks_for_no_verifier(
            self):
        argv = bootstrap_cmd(site="QQQQ", out=Path("b"),
                             args=self.parsed("--members", "12"))
        assert argv[argv.index("--stop-after") + 1] == "forecast"
        assert "--no-verify" in argv
        assert argv[argv.index("--members") + 1] == "12"

    def test_bootstrap_passes_a_drawn_box_through(self):
        argv = bootstrap_cmd(site="QQQQ", out=Path("b"),
                             args=self.parsed("--polygon", "box.json"))
        assert argv[argv.index("--domain-polygon") + 1] == "box.json"

    def test_the_daemon_defaults_to_the_hrrr_background(self):
        """HRRR is the background, permanently (Drew ruling, 2026-08-06).

        GFS stays selectable for archival reproduction of pre-HRRR runs
        and for nothing else; the daemon forwards whichever was chosen
        to the front door so every epoch's prepared case says so.
        """

        assert self.parsed().source == "hrrr"
        argv = bootstrap_cmd(site="QQQQ", out=Path("b"),
                             args=self.parsed())
        assert argv[argv.index("--source") + 1] == "hrrr"
        argv = bootstrap_cmd(site="QQQQ", out=Path("b"),
                             args=self.parsed("--source", "gfs"))
        assert argv[argv.index("--source") + 1] == "gfs"

    def test_bootstrap_asks_for_the_whole_epoch_of_boundary_data(self):
        argv = bootstrap_cmd(site="QQQQ", out=Path("b"),
                             args=self.parsed("--epoch-hours", "6"))
        assert argv[argv.index("--run-hours") + 1] == "6"

    def test_loop_argv_round_trips_through_the_parser(self):
        args = self.parsed("--members", "24", "--free-legs", "8")
        argv = loop_argv(args, Path("/run/root"))
        again = build_parser().parse_args(argv[3:])
        assert again.mode == "loop"
        assert again.members == 24
        assert again.free_legs == 8
        assert again.site == "QQQQ"

    def test_stop_and_status_only_need_the_directory(self):
        parsed = build_parser().parse_args(["stop", "--out", "o"])
        assert parsed.mode == "stop"
        assert build_parser().parse_args(
            ["status", "--out", "o"]).mode == "status"


# ---------------------------------------------------------------------------
# the status file is somebody else's poll target
# ---------------------------------------------------------------------------
class TestStatusFile:
    def test_round_trips(self, tmp_path):
        write_status(tmp_path, {"schema": SCHEMA, "state": "waiting"})
        assert read_status(tmp_path)["state"] == "waiting"

    def test_a_rewrite_leaves_no_temporary_behind(self, tmp_path):
        write_status(tmp_path, {"schema": SCHEMA, "state": "waiting"})
        write_status(tmp_path, {"schema": SCHEMA, "state": "stopped"})
        assert not (tmp_path / "auto-status.json.tmp").exists()
        assert read_status(tmp_path)["state"] == "stopped"

    def test_missing_status_says_nothing_started_here(self, tmp_path):
        with pytest.raises(AutoError, match="nothing has started"):
            read_status(tmp_path)

    def test_paths_are_where_the_launcher_expects_them(self, tmp_path):
        assert status_path(tmp_path).name == "auto-status.json"
        assert stop_path(tmp_path).name == "stop-requested"


# ---------------------------------------------------------------------------
# selection and the leg floor have to agree, or the daemon spins
# ---------------------------------------------------------------------------
class TestSelectionAgreesWithTheFloor:
    """A volume picked at exactly the floor can snap below it.

    Selection uses a raw time difference and the leg planner uses the
    snapped one, so without a step of slack a volume can be chosen and
    then refused by the same number -- and chosen again on the next
    poll, forever.  These pin the arithmetic that makes that impossible.
    """

    def test_without_slack_a_pick_can_snap_under_the_floor(self):
        dt, floor = 15.0, 900.0
        # a volume 900 s past the analysis, itself 7 s off the lattice
        vol = INIT + timedelta(seconds=907)
        assert (vol - INIT).total_seconds() >= floor       # selectable
        with pytest.raises(AutoError, match="sliver"):
            plan_leg(init=INIT, elapsed_s=7.0, volume_time=vol,
                     dt_s=dt, min_leg_s=floor)

    def test_a_step_of_slack_keeps_every_pick_plannable(self):
        dt, floor = 15.0, 900.0
        for offset in range(0, 15):
            vol = INIT + timedelta(seconds=floor + dt + offset)
            picked = pick_next_volume(
                [{"filename": "V", "key": "k",
                  "valid_time": vol.strftime("%Y-%m-%dT%H:%M:%SZ")}],
                after=INIT, min_gap_s=floor + dt)
            assert picked is not None
            plan = plan_leg(init=INIT, elapsed_s=0.0, volume_time=vol,
                            dt_s=dt, min_leg_s=floor)
            assert plan.leg_seconds >= floor


class TestCardCarriesThrough:
    """A run has to be sized against the card its caller was shown."""

    def parsed(self, *extra):
        return build_parser().parse_args(
            ["start", "--site", "qqqq", "--out", "o", *extra])

    def test_no_card_named_leaves_the_front_door_its_default(self):
        argv = bootstrap_cmd(site="QQQQ", out=Path("b"),
                             args=self.parsed())
        assert "--vram-gib" not in argv

    def test_a_named_card_reaches_the_front_door(self):
        argv = bootstrap_cmd(site="QQQQ", out=Path("b"),
                             args=self.parsed("--vram-gib", "31.84"))
        assert argv[argv.index("--vram-gib") + 1] == "31.84"

    def test_it_survives_the_start_to_loop_handoff(self):
        args = self.parsed("--vram-gib", "31.84")
        again = build_parser().parse_args(
            loop_argv(args, Path("/run/root"))[3:])
        assert again.vram_gib == 31.84


# ---------------------------------------------------------------------------
# a failure budget: retry a fault, stop for a wall
# ---------------------------------------------------------------------------
class TestFailureBudget:
    """A failed cycle does not advance the analysis clock.

    So retrying is right for a transient fault and wrong forever: every
    later volume eventually exceeds the leg ceiling and the daemon
    refuses each one in turn, busy and going nowhere.
    """

    def test_a_stage_failure_is_a_kind_of_refusal(self):
        from tools.da_nowcast_auto import StageFailure

        assert issubclass(StageFailure, AutoError)

    def test_a_data_refusal_is_not_a_stage_failure(self):
        from tools.da_nowcast_auto import StageFailure

        with pytest.raises(AutoError) as caught:
            plan_leg(init=INIT, elapsed_s=300.0,
                     volume_time=utc(2026, 8, 5, 4, 6), dt_s=15.0)
        assert not isinstance(caught.value, StageFailure)

    def test_a_child_that_fails_raises_a_stage_failure(self, tmp_path):
        from tools.da_nowcast_auto import StageFailure, run_step

        with pytest.raises(StageFailure, match="exit"):
            run_step("probe",
                     [sys.executable, "-c", "raise SystemExit(3)"],
                     cwd=tmp_path, log_dir=tmp_path, index=0)

    def test_the_budget_is_an_argument_with_a_stated_default(self):
        from tools.da_nowcast_auto import MAX_CONSECUTIVE_FAILURES

        args = build_parser().parse_args(
            ["start", "--site", "qqqq", "--out", "o"])
        assert args.max_consecutive_failures == MAX_CONSECUTIVE_FAILURES
        assert MAX_CONSECUTIVE_FAILURES >= 2

    def test_it_survives_the_start_to_loop_handoff(self):
        args = build_parser().parse_args(
            ["start", "--site", "qqqq", "--out", "o",
             "--max-consecutive-failures", "7"])
        again = build_parser().parse_args(
            loop_argv(args, Path("/run/root"))[3:])
        assert again.max_consecutive_failures == 7


class TestTerminalStatesRedraw:
    """A page that looks live and is not is the failure to avoid.

    Every way the daemon can stop routes through one place, so none of
    them can forget to leave the gallery saying it stopped.
    """

    def test_every_terminal_exit_goes_through_finish(self):
        import inspect

        from tools.da_nowcast_auto import Daemon

        source = inspect.getsource(Daemon.run)
        # the only bare returns left in the loop are the ones finish()
        # itself produces
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("return ") and "finish(" not in \
                    stripped and "self.finish" not in stripped:
                assert stripped in ("return 1", ), stripped

    def test_finish_publishes_the_state_it_was_given(self):
        from tools.da_nowcast_auto import Daemon

        assert "self.publish()" in __import__("inspect").getsource(
            Daemon.finish)

    def test_a_failed_render_does_not_swallow_the_stop(self):
        import inspect

        from tools.da_nowcast_auto import Daemon

        source = inspect.getsource(Daemon.finish)
        assert "except Exception" in source
        assert "return 1 if state ==" in source


class TestFailureExcerpt:
    """The tail of stderr is the least informative part of a crash.

    A CuPy process that dies mid-analysis floods stderr with excepthook
    cascade while it unwinds, so a receipt that kept only the last lines
    kept only the noise -- which is exactly what happened on the
    2026-08-05 launcher run and hid a plain out-of-memory error.
    """

    NOISE = "\n".join(["Original exception was:",
                       "Error in sys.excepthook:", ""] * 40)

    def test_the_first_traceback_wins_over_the_last_noise(self):
        from tools.da_nowcast_auto import failure_excerpt

        stderr = ("Traceback (most recent call last):\n"
                  "  File \"x.py\", line 1\n"
                  "MemoryError: out of memory\n" + self.NOISE)
        excerpt = failure_excerpt(stderr, "")
        assert "MemoryError: out of memory" in excerpt
        assert "sys.excepthook" not in excerpt

    def test_an_error_line_without_a_traceback_is_found(self):
        from tools.da_nowcast_auto import failure_excerpt

        excerpt = failure_excerpt(
            "some warning\nValueError: the real problem\n" + self.NOISE,
            "")
        assert "ValueError: the real problem" in excerpt

    def test_pure_noise_falls_back_to_stdout(self):
        from tools.da_nowcast_auto import failure_excerpt

        excerpt = failure_excerpt(self.NOISE, "leg 0 control: 1.8 s")
        assert "leg 0 control" in excerpt

    def test_nothing_at_all_still_says_something(self):
        from tools.da_nowcast_auto import failure_excerpt

        assert failure_excerpt("", "") == "(no output)"

    def test_the_receipt_keeps_both_ends(self, tmp_path):
        from tools.da_nowcast_auto import StageFailure, run_step

        script = ("import sys;"
                  r"[sys.stderr.write(f'line {i}\n') for i in range(300)];"
                  "sys.exit(1)")
        with pytest.raises(StageFailure):
            run_step("probe", [sys.executable, "-c", script],
                     cwd=tmp_path, log_dir=tmp_path, index=0)
        receipt = json.loads(
            (tmp_path / "00000-probe.json").read_text(encoding="utf-8"))
        assert receipt["stderr_head"][0] == "line 0"
        assert receipt["stderr_tail"][-1] == "line 299"
        assert receipt["stderr_lines"] == 300
