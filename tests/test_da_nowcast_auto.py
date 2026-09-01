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


# ===========================================================================
# EXPERIMENTAL: overlap handover -- the answer to an aging background
#
# The daemon's edge over an operational nowcast is CURRENCY, and a daemon
# left running holds one prepared case until its boundary data is spent:
# started on the 04Z background, still on the 04Z background at noon.
# The conservative fix is a second ensemble on a fresher case, spun up
# beside the first on the same volumes, promoted only once it has earned
# it.  Everything below is off unless --background-max-age is given.
# ===========================================================================
from tools.da_nowcast_auto import (                    # noqa: E402
    HANDOVER_COOLDOWN_MINUTES, HANDOVER_SCHEMA, HANDOVER_SPINUP_CYCLES,
    STATES, background_age_seconds, handover_due, handover_ready,
    handover_record, overlap_summary, repo_root, run_argument_dests)


class TestBackgroundAge:
    """The number the whole capability exists for."""

    def test_age_is_measured_from_the_case_init(self):
        assert background_age_seconds(
            init=INIT, now=INIT + timedelta(hours=3)) == 10800.0

    def test_it_is_not_the_analysis_age(self):
        # The cycling keeps the ANALYSIS current by construction, so an
        # age measured from it would read zero forever and the problem
        # would be invisible.  This is measured from the background.
        assert background_age_seconds(init=INIT, now=INIT) == 0.0
        assert background_age_seconds(
            init=INIT, now=INIT + timedelta(hours=8)) > 0.0


class TestHandoverDue:
    def test_off_by_default_is_off_at_any_age(self):
        assert not handover_due(init=INIT,
                                now=INIT + timedelta(days=3),
                                max_age_s=0.0)

    def test_a_young_background_is_left_alone(self):
        assert not handover_due(init=INIT,
                                now=INIT + timedelta(minutes=59),
                                max_age_s=3600.0)

    def test_the_threshold_is_inclusive(self):
        assert handover_due(init=INIT, now=INIT + timedelta(hours=1),
                            max_age_s=3600.0)

    def test_a_cooldown_holds_off_a_retry(self):
        now = INIT + timedelta(hours=4)
        assert not handover_due(init=INIT, now=now, max_age_s=3600.0,
                                cooldown_until=now + timedelta(minutes=1))
        assert handover_due(init=INIT, now=now, max_age_s=3600.0,
                            cooldown_until=now - timedelta(seconds=1))


class TestHandoverReady:
    """Two bars, and the second is the one that is easy to forget."""

    def test_a_spun_up_and_caught_up_ensemble_may_take_over(self):
        assert handover_ready(spinup_cycles=4, required_cycles=4,
                              spinup_analysis=utc(2026, 8, 5, 5, 54),
                              primary_analysis=utc(2026, 8, 5, 5, 54))

    def test_too_few_cycles_is_not_ready(self):
        assert not handover_ready(
            spinup_cycles=3, required_cycles=4,
            spinup_analysis=utc(2026, 8, 5, 5, 54),
            primary_analysis=utc(2026, 8, 5, 5, 54))

    def test_a_fresher_case_behind_the_running_analysis_is_refused(self):
        # A newer background whose analysis is half an hour behind is
        # not an improvement; it is a newer model of an older sky.
        assert not handover_ready(
            spinup_cycles=99, required_cycles=4,
            spinup_analysis=utc(2026, 8, 5, 5, 24),
            primary_analysis=utc(2026, 8, 5, 5, 54))

    def test_zero_required_cycles_still_needs_one(self):
        assert not handover_ready(
            spinup_cycles=0, required_cycles=0,
            spinup_analysis=INIT, primary_analysis=INIT)

    def test_the_default_bar_matches_the_measured_spin_up(self):
        # Analyses became competitive by cycle 3-4 on both live cases.
        assert 3 <= HANDOVER_SPINUP_CYCLES <= 4


class TestOverlapReceiptShapes:
    def test_a_handover_names_both_cases_and_the_window(self):
        record = handover_record(
            at=utc(2026, 8, 5, 6, 0), reason="aged out",
            retiring={"epoch": 0, "prepared_content_sha256": "old",
                      "cycles": 20},
            promoted={"epoch": 1, "prepared_content_sha256": "new",
                      "cycles": 8},
            overlap_started=utc(2026, 8, 5, 5, 30),
            primary_cycles_during_overlap=5)
        assert record["schema"] == HANDOVER_SCHEMA
        assert record["stability"] == "experimental"
        assert record["retired"]["prepared_content_sha256"] == "old"
        assert record["promoted"]["prepared_content_sha256"] == "new"
        assert record["overlap"]["seconds"] == 1800.0
        assert record["overlap"]["primary_cycles"] == 5
        assert record["overlap"]["spinup_cycles"] == 8
        assert "no ensemble state crossed cases" in record["note"]

    def test_the_summary_reports_age_even_with_the_capability_off(self):
        summary = overlap_summary(
            enabled=False, max_age_s=0.0, required_cycles=4,
            state="off", background_age_s=7200.0, spinup=None,
            cooldown_until=None, handovers=0)
        assert summary["enabled"] is False
        assert summary["state"] == "off"
        assert summary["background_age_seconds"] == 7200.0
        assert summary["background_max_age_seconds"] is None
        assert summary["stability"] == "experimental"

    def test_spinning_up_is_a_state_the_daemon_can_report(self):
        assert "spinning-up" in STATES


# ---------------------------------------------------------------------------
# THE TRAP: start rebuilds its own argv and spawns that
#
# A flag that parses but does not survive loop_argv is silently gone
# from the process that does the work, for the life of the daemon, with
# every receipt saying otherwise.  This walks the parser rather than a
# hand-written list, so a flag added without wiring fails HERE.
# ---------------------------------------------------------------------------
class TestReExecArgvSurvival:

    #: One non-default value per run flag.  A new flag with no entry
    #: here fails the coverage test below BY NAME -- which is the point:
    #: whoever adds it has to say what it should look like after the
    #: re-exec.
    PROBE = {
        "site": ["--site", "zzzz"],
        "out": None,                    # supplied per-test (tmp_path)
        "run_root": None,               # supplied per-test (repo root)
        "members": ["--members", "24"],
        "free_legs": ["--free-legs", "8"],
        "free_leg_seconds": ["--free-leg-seconds", "1200.0"],
        "dx_km": ["--dx-km", "1.5"],
        "box_half_km": ["--box-half-km", "99.5"],
        "polygon": None,                # supplied per-test (tmp_path)
        "physics_profile": ["--physics-profile", "probe-profile-v9"],
        "solve_device": ["--solve-device", "host"],
        "epoch_hours": ["--epoch-hours", "6"],
        "background_max_age": ["--background-max-age", "75.0"],
        "spinup_cycles": ["--spinup-cycles", "6"],
        "handover_cooldown_minutes": ["--handover-cooldown-minutes",
                                      "12.5"],
        "poll_seconds": ["--poll-seconds", "17"],
        "render_every": ["--render-every", "3"],
        "min_leg_seconds": ["--min-leg-seconds", "150.0"],
        "max_leg_seconds": ["--max-leg-seconds", "1500.0"],
        "max_cycles": ["--max-cycles", "42"],
        "max_epochs": ["--max-epochs", "9"],
        "max_consecutive_failures": ["--max-consecutive-failures", "7"],
        "vram_gib": ["--vram-gib", "31.84"],
        "bucket": ["--bucket", "probe-bucket"],
        "geog_root": None,              # supplied per-test (tmp_path)
        "bridge": None,                 # supplied per-test (tmp_path)
        "allow_stale": ["--allow-stale"],
        "horizontal_loc_m": ["--horizontal-loc-m", "9000.0"],
        "vertical_loc_m": ["--vertical-loc-m", "2500.0"],
        "length_scale_km": ["--length-scale-km", "45.0"],
        "seed": ["--seed", "1234"],
        # Not the parser default, so a re-exec that dropped the flag and
        # fell back to it fails here instead of preparing every later
        # epoch on a background nobody asked for.
        "source": ["--source", "hrrr"],
        "dealias": ["--dealias"],
        "dealias_engine": ["--dealias-engine", "vad-region"],
        # A PAIR of flags, not a valued option, and refinement is
        # refused on this engine -- so the probe states the negative,
        # which is the value that must survive the roll unchanged.
        "dealias_refinement": ["--no-dealias-refinement"],
    }

    PATHY = {"out", "polygon", "geog_root", "bridge", "run_root"}

    def probe_argv(self, tmp_path):
        argv = ["start"]
        for value in self.PROBE.values():
            if value is not None:
                argv.extend(value)
        argv.extend(("--out", str(tmp_path / "daemon")))
        argv.extend(("--polygon", str(tmp_path / "box.json")))
        argv.extend(("--geog-root", str(tmp_path / "geog")))
        argv.extend(("--bridge", str(tmp_path / "bridge.exe")))
        argv.extend(("--run-root", str(repo_root())))
        return argv

    def test_every_run_flag_has_a_probe(self):
        missing = sorted(set(run_argument_dests()) - set(self.PROBE))
        assert not missing, (
            f"{missing} is defined in add_run_arguments but has no "
            "entry here. Add it to loop_argv AND to PROBE: a flag that "
            "does not survive the start-to-loop re-exec is silently "
            "gone from the daemon that does the work.")

    def test_no_probe_names_a_flag_that_no_longer_exists(self):
        assert not sorted(set(self.PROBE) - set(run_argument_dests()))

    def test_every_run_flag_survives_the_re_exec(self, tmp_path):
        args = build_parser().parse_args(self.probe_argv(tmp_path))
        rebuilt = loop_argv(args, repo_root())
        assert rebuilt[3] == "loop"
        again = build_parser().parse_args(rebuilt[3:])

        def norm(dest, value):
            if dest in self.PATHY and value is not None:
                return Path(value).resolve()
            return value

        for dest in run_argument_dests():
            assert norm(dest, getattr(again, dest)) == norm(
                dest, getattr(args, dest)), (
                f"--{dest.replace('_', '-')} did not survive the "
                "start-to-loop re-exec")

    def test_the_overlap_flags_specifically_survive(self, tmp_path):
        args = build_parser().parse_args(self.probe_argv(tmp_path))
        again = build_parser().parse_args(
            loop_argv(args, repo_root())[3:])
        assert again.background_max_age == 75.0
        assert again.spinup_cycles == 6
        assert again.handover_cooldown_minutes == 12.5

    def test_the_capability_is_off_when_nobody_asked_for_it(self):
        args = build_parser().parse_args(
            ["start", "--site", "qqqq", "--out", "o"])
        assert args.background_max_age == 0.0
        assert args.spinup_cycles == HANDOVER_SPINUP_CYCLES
        assert (args.handover_cooldown_minutes
                == HANDOVER_COOLDOWN_MINUTES)

    def test_off_survives_the_re_exec_as_off(self, tmp_path):
        args = build_parser().parse_args(
            ["start", "--site", "qqqq", "--out", str(tmp_path)])
        again = build_parser().parse_args(
            loop_argv(args, repo_root())[3:])
        assert again.background_max_age == 0.0


# ---------------------------------------------------------------------------
# The orchestration itself, on a fake clock and a fake card.
#
# No GPU, no network, no child process: ``run_step`` is replaced by a
# recorder that writes back the one artefact the daemon reads (a cycle
# report), so the REAL cycle body, the REAL loop and the REAL handover
# run.  What is faked is the card and the calendar, not the logic.
# ---------------------------------------------------------------------------
import tools.da_nowcast_auto as auto                   # noqa: E402
from tools.da_nowcast import parse_iso                 # noqa: E402
from tools.da_nowcast_auto import (                    # noqa: E402
    RETIRED_SCHEMA, Daemon, Lane, StageFailure)


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


class FakeCard:
    """Every child process the daemon would launch, as a record.

    One at a time, in order, because the daemon runs them from one
    thread -- which is precisely the admission the real card gets and
    the reason an overlap costs a second cycle rather than a second
    process on the device.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.fail_when = None           # (name, argv) -> message | None

    @property
    def stages(self) -> list[str]:
        return [name for name, _ in self.calls]

    def cycle_argvs(self) -> list[list[str]]:
        return [argv for name, argv in self.calls
                if name.startswith("cycle-")]

    def render_case_dirs(self) -> list[str]:
        return [argv[argv.index("--case-dir") + 1]
                for name, argv in self.calls
                if name.startswith("render-")]

    def __call__(self, name, argv, *, cwd, log_dir, index):
        self.calls.append((name, list(argv)))
        if self.fail_when is not None:
            message = self.fail_when(name, argv)
            if message:
                raise StageFailure(f"{name} failed (exit 1): {message}")
        if name.startswith("cycle-"):
            out = Path(argv[-1])
            free = int(argv[argv.index("--free-legs") + 1])
            (out / "composites").mkdir(parents=True, exist_ok=True)
            legs = [{"leg": 0, "analysis": {"solve_seconds": 1.5}}]
            legs += [{"leg": k, "analysis": None}
                     for k in range(1, free + 1)]
            (out / "cycle-report.json").write_text(json.dumps(
                {"schema": "gpuwm-da.cycle-prepared.v1",
                 "args": {"members": 10}, "legs": legs}),
                encoding="utf-8")
        return {"stage": name, "returncode": 0}


def fake_epoch(number: int, role: str, init: datetime, root: Path
               ) -> dict:
    base = root / "epochs" / f"epoch{number:04d}"
    return {
        "number": number, "role": role, "init": iso_z(init),
        "run_seconds": 14400.0, "dt_s": 15.0,
        "history_interval_s": 900.0,
        "nx": 132, "ny": 132, "nz": 45,
        "case_name": f"probe-case-{number:04d}",
        "bootstrap": str(base / "bootstrap"),
        "authority": str(base / "authority"),
        "prepared_root": str(base / "prepared"),
        "run_dir": str(base / "run"),
        "proof_sha256": f"proof{number:04d}",
        "source_manifest_sha256": f"manifest{number:04d}",
        # The binding the whole capability exists to respect: one
        # digest per prepared case, and no state crosses between them.
        "prepared_content_sha256": f"content{number:04d}" + "0" * 50,
        "seed": 20260805, "elapsed_seconds": 0.0, "cycles": 0,
        "length_scale_km": 40.0, "length_scale_note": "",
        # The real epoch writer carries this from the bindings receipt,
        # which is the authority on which background the prepared case
        # is.  Deliberately NOT the argparse default, so a regression to
        # a hardcoded or defaulted source is a failure here rather than a
        # cycle that quietly reports the wrong background.
        "source": "hrrr",
    }


def iso_z(stamp: datetime) -> str:
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def georeference(lane: Lane, *, hours: int = 4) -> None:
    """The wrfouts a cycle grids its observations onto."""

    run_dir = Path(lane.epoch["run_dir"]) / "wrfout"
    run_dir.mkdir(parents=True, exist_ok=True)
    for step in range(int(hours * 4) + 1):
        stamp = lane.init + timedelta(seconds=900 * step)
        (run_dir / f"wrfout_d01_{stamp:%Y-%m-%d_%H_%M_%S}").write_text(
            "georeference", encoding="utf-8")


class Rig:
    """A daemon with a fake card, a fake clock and a fake archive."""

    def __init__(self, tmp_path, monkeypatch, *, inits, volumes,
                 now, extra=()):
        self.card = FakeCard()
        self.clock = FakeClock(now)
        self.volumes = volumes
        self.inits = list(inits)
        self.built: list[Lane] = []
        self.listings = 0
        self.notices: list[dict] = []

        rig = self

        def build_lane(daemon, *, role):
            return rig.build_lane(daemon, role=role)

        announced = Daemon.announce

        def announce(daemon, level, headline, detail):
            rig.notices.append({"level": level, "headline": headline,
                                "detail": detail})
            return announced(daemon, level, headline, detail)

        monkeypatch.setattr(auto, "now_utc", self.clock)
        monkeypatch.setattr(auto, "run_step", self.card)
        monkeypatch.setattr(auto, "worktree_fingerprint",
                            lambda root: {"is_git": False, "head": "",
                                          "dirty_paths": 0})
        monkeypatch.setattr(auto, "list_site_volumes", self.listing)
        monkeypatch.setattr(auto.ens_state, "latest_generation",
                            lambda root, **kw: (Path(root) / "slot00",
                                                {"leg_number": 0}))
        monkeypatch.setattr(auto.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(Daemon, "build_lane", build_lane)
        monkeypatch.setattr(Daemon, "announce", announce)

        args = auto.build_parser().parse_args(
            ["loop", "--site", "qqqq",
             "--out", str(tmp_path / "daemon"),
             "--run-root", str(auto.repo_root()), *extra])
        self.daemon = Daemon(args)

    def headlines(self) -> list[str]:
        return [notice["headline"] for notice in self.notices]

    def said(self, prefix: str) -> int:
        for index, headline in enumerate(self.headlines()):
            if headline.startswith(prefix):
                return index
        raise AssertionError(f"nothing said {prefix!r}; said "
                             f"{self.headlines()}")

    def listing(self, *, site, start, end, bucket):
        self.listings += 1
        assert self.listings < 400, "the loop is not terminating"
        return [v for v in self.volumes
                if start <= parse_iso(v["valid_time"]) <= end]

    def build_lane(self, daemon, *, role):
        daemon.epoch_number += 1
        number = daemon.epoch_number
        init = self.inits.pop(0)
        self.card(f"bootstrap-{number:04d}", ["--role", role],
                  cwd=daemon.run_root,
                  log_dir=daemon.out / "epochs", index=0)
        lane = Lane(number=number, role=role,
                    epoch=fake_epoch(number, role, init, daemon.out),
                    root=daemon.out, started=self.clock())
        georeference(lane)
        self.built.append(lane)
        return lane

    def install_primary(self, *, init, elapsed_s=0.0, cycles=0):
        lane = self.build_lane(self.daemon, role="primary")
        assert lane.init == init, "inits queue is out of order"
        lane.epoch["elapsed_seconds"] = elapsed_s
        lane.epoch["cycles"] = cycles
        self.daemon.primary = lane
        return lane

    def install_spinup(self, *, elapsed_s=0.0, cycles=0):
        lane = self.build_lane(self.daemon, role="spinup")
        lane.epoch["elapsed_seconds"] = elapsed_s
        lane.epoch["cycles"] = cycles
        self.daemon.spinup = lane
        self.daemon.overlap_started = self.clock()
        return lane


def feed(first: datetime, count: int, *, every_s: int = 360
         ) -> list[dict]:
    return [{"filename": f"VOL{index:03d}", "key": f"k/{index:03d}",
             "valid_time": iso_z(first + timedelta(seconds=every_s
                                                   * index))}
            for index in range(count)]


class TestOverlapHandoverEndToEnd:
    """One daemon, two ensembles, one gallery, one switch."""

    def rig(self, tmp_path, monkeypatch):
        # Primary on the 05:00Z background; the feed runs to 05:54Z; it
        # is 06:30Z, so the running background is 90 minutes old.
        volumes = feed(utc(2026, 8, 5, 5, 6), 9)
        return Rig(tmp_path, monkeypatch,
                   inits=[utc(2026, 8, 5, 5, 0), utc(2026, 8, 5, 5, 30)],
                   volumes=volumes, now=utc(2026, 8, 5, 6, 30),
                   extra=["--background-max-age", "60",
                          # Deliberately BELOW what the spin-up needs to
                          # catch up, so the caught-up bar is what
                          # decides the moment and the test can see it.
                          "--spinup-cycles", "2",
                          "--free-legs", "2", "--render-every", "4"])

    def run_to_handover(self, rig):
        rig.daemon.stop_requested = lambda: bool(rig.daemon.handovers)
        return rig.daemon.run()

    def test_the_gallery_switches_to_the_fresher_case(self, tmp_path,
                                                      monkeypatch):
        rig = self.rig(tmp_path, monkeypatch)
        assert self.run_to_handover(rig) == 0
        daemon = rig.daemon
        assert len(daemon.handovers) == 1
        assert daemon.primary.number == 1
        assert daemon.primary.role == "primary"
        assert daemon.spinup is None
        assert daemon.primary.init == utc(2026, 8, 5, 5, 30)

    def test_the_two_lanes_are_different_prepared_cases(self, tmp_path,
                                                        monkeypatch):
        rig = self.rig(tmp_path, monkeypatch)
        self.run_to_handover(rig)
        record = rig.daemon.handovers[0]
        assert (record["retired"]["prepared_content_sha256"]
                != record["promoted"]["prepared_content_sha256"])
        assert record["retired"]["epoch"] == 0
        assert record["promoted"]["epoch"] == 1
        assert record["promoted"]["cycles"] >= 2

    def test_the_spin_up_never_ran_a_free_forecast(self, tmp_path,
                                                   monkeypatch):
        rig = self.rig(tmp_path, monkeypatch)
        self.run_to_handover(rig)
        spun = [argv for argv in rig.card.cycle_argvs()
                if "epoch0001" in argv[-1]]
        assert spun, "the spin-up lane never cycled"
        for argv in spun:
            assert argv[argv.index("--free-legs") + 1] == "0", (
                "a spin-up cycle paid for a free forecast nobody was "
                "being shown")

    def test_it_handed_over_only_once_caught_up(self, tmp_path,
                                               monkeypatch):
        rig = self.rig(tmp_path, monkeypatch)
        self.run_to_handover(rig)
        record = rig.daemon.handovers[0]
        assert record["promoted"]["analysis"] >= record[
            "retired"]["analysis"]
        # --spinup-cycles was 2 and it took 4, because the first two
        # cycles still left the fresh analysis behind the running one.
        # A fresher background on an older sky is not an improvement,
        # so the caught-up bar, not the cycle count, set the moment.
        assert record["promoted"]["cycles"] == 4, (
            "the caught-up bar did not delay the handover")

    def test_the_history_says_which_ensemble_each_cycle_came_from(
            self, tmp_path, monkeypatch):
        rig = self.rig(tmp_path, monkeypatch)
        self.run_to_handover(rig)
        lanes = {entry["lane"] for entry in rig.daemon.history}
        assert lanes == {"primary", "spinup"}
        digests = {entry["lane"]: entry["case_digest"]
                   for entry in rig.daemon.history}
        assert digests["primary"] != digests["spinup"]

    def test_the_handover_is_written_where_it_survives_the_daemon(
            self, tmp_path, monkeypatch):
        rig = self.rig(tmp_path, monkeypatch)
        self.run_to_handover(rig)
        path = rig.daemon.out / "overlap-handovers.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == HANDOVER_SCHEMA
        assert payload["stability"] == "experimental"
        assert len(payload["handovers"]) == 1
        assert payload["handovers"][0]["overlap"]["spinup_cycles"] == 4

    def test_the_retired_lane_is_marked_and_nothing_is_deleted(
            self, tmp_path, monkeypatch):
        rig = self.rig(tmp_path, monkeypatch)
        self.run_to_handover(rig)
        old = rig.daemon.out / "epochs" / "epoch0000"
        marker = json.loads(
            (old / "retired.json").read_text(encoding="utf-8"))
        assert marker["schema"] == RETIRED_SCHEMA
        assert "nothing was deleted" in marker["note"]
        # Its cycles, its view and its georeference are all still there.
        assert (old / "cycles").is_dir()
        assert (old / "view" / "cycle" / "cycle-report.json").is_file()
        assert list((old / "run" / "wrfout").glob("wrfout_d01_*"))

    def test_the_gallery_is_drawn_from_the_new_case_after_the_switch(
            self, tmp_path, monkeypatch):
        rig = self.rig(tmp_path, monkeypatch)
        self.run_to_handover(rig)
        drawn = rig.card.render_case_dirs()
        assert "epoch0000" in drawn[0]
        assert "epoch0001" in drawn[-1]
        assert str(rig.daemon.view_dir()).endswith(
            str(Path("epoch0001") / "view"))

    def test_the_switch_is_one_honest_line_on_the_gallery(
            self, tmp_path, monkeypatch):
        rig = self.rig(tmp_path, monkeypatch)
        self.run_to_handover(rig)
        said = rig.said("switched to a 05:30Z background (was 05:00Z)")
        notice = rig.notices[said]
        assert notice["level"] == "info"
        assert "no ensemble state crossed" in notice["detail"]
        assert "kept on disk" in notice["detail"]
        # Said AFTER the swap, so it is written into the view the very
        # next render draws: the first frame off the new case is the
        # frame that says it is a new case.
        drawn = [index for index, (name, _) in enumerate(rig.card.calls)
                 if name.startswith("render-")]
        assert (rig.daemon.view_dir() / "auto-notice.json").is_file()
        assert rig.card.render_case_dirs()[-1].endswith(
            str(Path("epoch0001") / "view"))
        assert drawn, "the gallery was never redrawn"

    def test_the_preparation_is_announced_before_it_costs_anything(
            self, tmp_path, monkeypatch):
        rig = self.rig(tmp_path, monkeypatch)
        self.run_to_handover(rig)
        assert rig.said("preparing a fresher background") < rig.said(
            "spinning up a 05:30Z background")
        assert rig.said("spinning up a 05:30Z background") < rig.said(
            "switched to a 05:30Z")

    def test_the_status_file_carries_the_overlap(self, tmp_path,
                                                 monkeypatch):
        rig = self.rig(tmp_path, monkeypatch)
        self.run_to_handover(rig)
        status = read_status(rig.daemon.out)
        assert status["overlap"]["enabled"] is True
        assert status["overlap"]["handovers"] == 1
        assert status["overlap"]["spinup_cycles_required"] == 2
        assert status["background_age_seconds"] == 3600.0
        assert status["retired_lanes"][0]["epoch"] == 0


class TestDefaultIsExactlyTheOldBehaviour:
    """Off is off: an ancient background, and not one extra case."""

    def test_no_second_lane_is_ever_prepared(self, tmp_path,
                                             monkeypatch):
        volumes = feed(utc(2026, 8, 5, 5, 6), 9)
        rig = Rig(tmp_path, monkeypatch,
                  inits=[utc(2026, 8, 5, 5, 0)], volumes=volumes,
                  now=utc(2026, 8, 5, 18, 0),      # 13 hours old
                  extra=["--free-legs", "2"])
        rig.daemon.stop_requested = lambda: rig.daemon.cycles >= 9
        assert rig.daemon.run() == 0
        assert rig.daemon.spinup is None
        assert rig.daemon.handovers == []
        assert rig.daemon.epoch_number == 0
        assert sorted(p.name for p in
                      (rig.daemon.out / "epochs").iterdir()) == [
            "epoch0000"]

    def test_the_age_is_still_reported(self, tmp_path, monkeypatch):
        rig = Rig(tmp_path, monkeypatch,
                  inits=[utc(2026, 8, 5, 5, 0)],
                  volumes=feed(utc(2026, 8, 5, 5, 6), 2),
                  now=utc(2026, 8, 5, 18, 0))
        rig.install_primary(init=utc(2026, 8, 5, 5, 0))
        rig.daemon.publish()
        status = read_status(rig.daemon.out)
        assert status["overlap"]["enabled"] is False
        assert status["overlap"]["state"] == "off"
        assert status["background_age_seconds"] == 46800.0


class TestFailSafe:
    """Freshness is never worth killing a working nowcast for."""

    def rig(self, tmp_path, monkeypatch, *, extra=()):
        rig = Rig(tmp_path, monkeypatch,
                  inits=[utc(2026, 8, 5, 5, 0), utc(2026, 8, 5, 5, 30)],
                  volumes=feed(utc(2026, 8, 5, 5, 6), 9),
                  now=utc(2026, 8, 5, 6, 30),
                  extra=["--background-max-age", "60",
                         "--spinup-cycles", "4", *extra])
        rig.install_primary(init=utc(2026, 8, 5, 5, 0),
                            elapsed_s=3240.0, cycles=9)
        return rig

    def test_a_case_that_will_not_prepare_leaves_the_daemon_cycling(
            self, tmp_path, monkeypatch):
        rig = self.rig(tmp_path, monkeypatch)
        running = rig.daemon.primary
        rig.card.fail_when = (
            lambda name, argv: "the fetch fell over"
            if name.startswith("bootstrap") else None)

        assert rig.daemon.maybe_start_spinup(
            rig.clock(), primary_current=True) is True
        assert rig.daemon.spinup is None
        assert rig.daemon.primary is running, (
            "the running ensemble was disturbed by a failed handover")
        assert rig.daemon.state != "failed"
        assert rig.daemon.failures == 0, (
            "a spin-up failure spent the primary's failure budget")
        assert rig.daemon.notice["level"] == "warn"
        assert "stayed on the current background" in (
            rig.daemon.notice["headline"])
        assert "the fetch fell over" in rig.daemon.notice["detail"]

    def test_the_failure_starts_a_cooldown_rather_than_a_retry_storm(
            self, tmp_path, monkeypatch):
        rig = self.rig(tmp_path, monkeypatch,
                       extra=["--handover-cooldown-minutes", "20"])
        rig.card.fail_when = (
            lambda name, argv: "nope"
            if name.startswith("bootstrap") else None)
        rig.daemon.maybe_start_spinup(rig.clock(), primary_current=True)
        assert rig.daemon.cooldown_until == rig.clock() + timedelta(
            minutes=20)
        # And the next pass declines to try again.
        before = len(rig.card.calls)
        assert rig.daemon.maybe_start_spinup(
            rig.clock(), primary_current=True) is False
        assert len(rig.card.calls) == before
        assert rig.daemon.overlap_state() == "cooldown"

    def test_a_spin_up_that_will_not_cycle_is_abandoned_not_escalated(
            self, tmp_path, monkeypatch):
        rig = self.rig(tmp_path, monkeypatch)
        running = rig.daemon.primary
        spinning = rig.install_spinup()
        rig.card.fail_when = (
            lambda name, argv: "the solve fell over"
            if (name.startswith("cycle-") and "epoch0001" in argv[-1])
            else None)
        budget = rig.daemon.args.max_consecutive_failures
        for _ in range(budget):
            rig.daemon.advance_spinup(rig.volumes)
        assert rig.daemon.spinup is None
        assert rig.daemon.primary is running
        assert rig.daemon.failures == 0
        assert rig.daemon.state != "failed"
        assert "stayed on the current background" in (
            rig.daemon.notice["headline"])
        marker = json.loads((spinning.dir / "retired.json")
                            .read_text(encoding="utf-8"))
        assert "failed" in marker["reason"]
        assert "nothing was deleted" in marker["note"]

    def test_a_case_that_is_not_actually_fresher_is_not_handed_over_to(
            self, tmp_path, monkeypatch):
        # A bootstrap's init is the most recent whole hour at or before
        # the newest volume.  Ask for a fresher background more often
        # than the archive publishes one and you are handed back the
        # SAME background -- which, unguarded, would hand over to it,
        # find the new lane just as old, and prepare another forever.
        rig = Rig(tmp_path, monkeypatch,
                  inits=[utc(2026, 8, 5, 5, 0), utc(2026, 8, 5, 5, 0)],
                  volumes=feed(utc(2026, 8, 5, 5, 6), 9),
                  now=utc(2026, 8, 5, 5, 50),
                  extra=["--background-max-age", "45",
                         "--spinup-cycles", "2"])
        running = rig.install_primary(init=utc(2026, 8, 5, 5, 0),
                                      elapsed_s=2520.0, cycles=7)

        assert rig.daemon.maybe_start_spinup(
            rig.clock(), primary_current=True) is True
        assert rig.daemon.spinup is None
        assert rig.daemon.primary is running
        assert rig.daemon.handovers == []
        assert rig.daemon.cooldown_until is not None
        assert "nothing fresher" in rig.daemon.notice["detail"]
        # And it does not immediately try again.
        assert rig.daemon.maybe_start_spinup(
            rig.clock(), primary_current=True) is False

    def test_one_bad_spin_up_cycle_is_not_the_end_of_the_spin_up(
            self, tmp_path, monkeypatch):
        rig = self.rig(tmp_path, monkeypatch)
        rig.install_spinup()
        rig.card.fail_when = (
            lambda name, argv: "transient"
            if (name.startswith("cycle-") and "epoch0001" in argv[-1])
            else None)
        rig.daemon.advance_spinup(rig.volumes)
        assert rig.daemon.spinup is not None
        assert rig.daemon.spinup_failures == 1

    def test_the_daemon_survives_a_spin_up_that_never_works(
            self, tmp_path, monkeypatch):
        # The whole loop, not one method: a handover that cannot happen
        # must not stop the nowcast that can.
        rig = Rig(tmp_path, monkeypatch,
                  inits=[utc(2026, 8, 5, 5, 0), utc(2026, 8, 5, 5, 30)],
                  volumes=feed(utc(2026, 8, 5, 5, 6), 9),
                  now=utc(2026, 8, 5, 6, 30),
                  extra=["--background-max-age", "60",
                         "--spinup-cycles", "4", "--free-legs", "2"])
        rig.card.fail_when = (
            lambda name, argv: "the fresh case will not build"
            if name.startswith("bootstrap-0001") else None)
        rig.daemon.stop_requested = lambda: (
            rig.daemon.cooldown_until is not None
            and rig.daemon.cycles >= 9)
        assert rig.daemon.run() == 0
        assert rig.daemon.state == "stopped"
        assert rig.daemon.cycles == 9, (
            "the primary stopped cycling because a handover failed")
        assert rig.daemon.handovers == []
        assert rig.daemon.primary.number == 0


class TestExhaustedCasePromotesRatherThanColdStarts:
    """A spent case with a spin-up beside it: use the spin-up."""

    def test_a_partly_spun_up_lane_beats_a_cold_bootstrap(
            self, tmp_path, monkeypatch):
        rig = Rig(tmp_path, monkeypatch,
                  inits=[utc(2026, 8, 5, 5, 0), utc(2026, 8, 5, 5, 30)],
                  volumes=feed(utc(2026, 8, 5, 5, 6), 9),
                  now=utc(2026, 8, 5, 6, 30),
                  extra=["--background-max-age", "60",
                         "--spinup-cycles", "4"])
        rig.install_primary(init=utc(2026, 8, 5, 5, 0),
                            elapsed_s=14000.0, cycles=30)
        rig.install_spinup(elapsed_s=1440.0, cycles=2)
        assert rig.daemon.needs_new_epoch()
        rig.daemon.roll_primary()
        assert rig.daemon.primary.number == 1
        assert rig.daemon.epoch_number == 1, (
            "a third case was prepared when one was already spun up")
        record = rig.daemon.handovers[0]
        assert "boundary data is spent" in record["reason"]
        assert "2 of 4 cycles" in record["reason"], (
            "the receipt hid that this handover was forced early")

    def test_with_no_spin_up_it_cold_starts_exactly_as_before(
            self, tmp_path, monkeypatch):
        rig = Rig(tmp_path, monkeypatch,
                  inits=[utc(2026, 8, 5, 5, 0), utc(2026, 8, 5, 6, 0)],
                  volumes=feed(utc(2026, 8, 5, 5, 6), 9),
                  now=utc(2026, 8, 5, 6, 30))
        rig.install_primary(init=utc(2026, 8, 5, 5, 0),
                            elapsed_s=14000.0, cycles=30)
        rig.daemon.roll_primary()
        assert rig.daemon.primary.number == 1
        assert rig.daemon.handovers == []
        assert rig.daemon.retired == []
