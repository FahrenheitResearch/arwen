"""Unit tests for the delayed-window multi-storm scorecard harness.

Pure-logic coverage only: run finding, day-boundary merging, case
window geometry (including the delay floor that IS the harness's
claim), deduplication, aggregation, and the frozen-config hash.  No
network, no GPU, no radar data.  Site ids here are synthetic
four-letter tokens, never real stations (standing owner rule).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.da_scorecard import (BASELINE_CONFIG, DELAY_FLOOR_SECONDS,
                                aggregate_frames, apply_max_cases,
                                campaign_verdict, case_is_closed,
                                case_window, config_hash, dedup_cases,
                                front_door_argv, haversine_km,
                                lead_minutes, merge_runs, place_window,
                                precip_runs, skip_reason)

UTC = timezone.utc


def volume(stamp: str, size: int) -> dict:
    return {"valid_time": stamp, "size_bytes": size}


def ramp(start: datetime, count: int, *, step_s: float,
         size: int) -> list[dict]:
    return [volume((start + timedelta(seconds=i * step_s))
                   .strftime("%Y-%m-%dT%H:%M:%SZ"), size)
            for i in range(count)]


class TestPrecipRuns:
    def test_finds_one_sustained_run(self):
        t0 = datetime(2026, 1, 7, 2, 0, tzinfo=UTC)
        vols = ramp(t0, 30, step_s=300, size=14_000_000)
        runs = precip_runs(vols, min_size_bytes=8_000_000,
                           max_cadence_seconds=420,
                           min_duration_seconds=5400)
        assert len(runs) == 1
        assert runs[0]["volumes"] == 30
        assert runs[0]["duration_seconds"] == pytest.approx(29 * 300)

    def test_small_volumes_never_qualify(self):
        t0 = datetime(2026, 1, 7, 2, 0, tzinfo=UTC)
        vols = ramp(t0, 30, step_s=300, size=3_000_000)
        assert precip_runs(vols, min_size_bytes=8_000_000,
                           max_cadence_seconds=420,
                           min_duration_seconds=5400) == []

    def test_cadence_gap_splits_the_run(self):
        t0 = datetime(2026, 1, 7, 2, 0, tzinfo=UTC)
        vols = (ramp(t0, 12, step_s=300, size=14_000_000)
                + ramp(t0 + timedelta(hours=3), 12, step_s=300,
                       size=14_000_000))
        runs = precip_runs(vols, min_size_bytes=8_000_000,
                           max_cadence_seconds=420,
                           min_duration_seconds=1800)
        assert len(runs) == 2

    def test_short_run_is_dropped(self):
        t0 = datetime(2026, 1, 7, 2, 0, tzinfo=UTC)
        vols = ramp(t0, 4, step_s=300, size=14_000_000)
        assert precip_runs(vols, min_size_bytes=8_000_000,
                           max_cadence_seconds=420,
                           min_duration_seconds=5400) == []

    def test_unsorted_input_is_sorted_first(self):
        t0 = datetime(2026, 1, 7, 2, 0, tzinfo=UTC)
        vols = ramp(t0, 20, step_s=300, size=14_000_000)
        runs_fwd = precip_runs(list(vols), min_size_bytes=8_000_000,
                               max_cadence_seconds=420,
                               min_duration_seconds=3600)
        runs_rev = precip_runs(list(reversed(vols)),
                               min_size_bytes=8_000_000,
                               max_cadence_seconds=420,
                               min_duration_seconds=3600)
        assert runs_fwd == runs_rev


class TestMergeRuns:
    def test_midnight_split_becomes_one_event(self):
        runs = [
            {"start": "2026-01-06T22:00:00Z", "end": "2026-01-06T23:57:00Z",
             "volumes": 24, "median_size_bytes": 14_000_000,
             "duration_seconds": 7020.0},
            {"start": "2026-01-07T00:02:00Z", "end": "2026-01-07T03:00:00Z",
             "volumes": 36, "median_size_bytes": 15_000_000,
             "duration_seconds": 10680.0},
        ]
        merged = merge_runs(runs, max_gap_seconds=420)
        assert len(merged) == 1
        assert merged[0]["start"] == "2026-01-06T22:00:00Z"
        assert merged[0]["end"] == "2026-01-07T03:00:00Z"
        assert merged[0]["volumes"] == 60
        assert merged[0]["duration_seconds"] == pytest.approx(18000.0)

    def test_distinct_events_stay_distinct(self):
        runs = [
            {"start": "2026-01-06T20:00:00Z", "end": "2026-01-06T22:00:00Z",
             "volumes": 24, "median_size_bytes": 14_000_000,
             "duration_seconds": 7200.0},
            {"start": "2026-01-07T04:00:00Z", "end": "2026-01-07T06:00:00Z",
             "volumes": 24, "median_size_bytes": 14_000_000,
             "duration_seconds": 7200.0},
        ]
        assert len(merge_runs(runs, max_gap_seconds=420)) == 2

    def test_input_order_does_not_matter(self):
        runs = [
            {"start": "2026-01-07T00:02:00Z", "end": "2026-01-07T03:00:00Z",
             "volumes": 36, "median_size_bytes": 15_000_000,
             "duration_seconds": 10680.0},
            {"start": "2026-01-06T22:00:00Z", "end": "2026-01-06T23:57:00Z",
             "volumes": 24, "median_size_bytes": 14_000_000,
             "duration_seconds": 7020.0},
        ]
        assert len(merge_runs(runs, max_gap_seconds=420)) == 1


class TestCaseWindow:
    NOW = datetime(2026, 1, 8, 12, 0, tzinfo=UTC)

    def kwargs(self, **over):
        base = dict(cycles=6, cycle_seconds=900, free_legs=6,
                    now=self.NOW)
        base.update(over)
        return base

    def test_whole_hour_init_and_covered_cycling(self):
        init, window_end = case_window(
            datetime(2026, 1, 7, 3, 40, tzinfo=UTC),
            datetime(2026, 1, 7, 7, 0, tzinfo=UTC), **self.kwargs())
        assert init == datetime(2026, 1, 7, 4, 0, tzinfo=UTC)
        assert window_end == datetime(2026, 1, 7, 5, 30, tzinfo=UTC)

    def test_exact_hour_start_is_kept(self):
        init, _ = case_window(
            datetime(2026, 1, 7, 4, 0, tzinfo=UTC),
            datetime(2026, 1, 7, 7, 0, tzinfo=UTC), **self.kwargs())
        assert init == datetime(2026, 1, 7, 4, 0, tzinfo=UTC)

    def test_run_too_short_for_cycling_refuses(self):
        got, why = case_window(
            datetime(2026, 1, 7, 3, 40, tzinfo=UTC),
            datetime(2026, 1, 7, 5, 0, tzinfo=UTC), **self.kwargs())
        assert got is None
        assert "outlive the storm" in why

    def test_delay_floor_refuses_an_unclosed_case(self):
        # Free legs end 11:30; the 90-minute floor is not cleared at
        # 12:00, and an unclosed case would mean unverifiable frames.
        got, why = case_window(
            datetime(2026, 1, 8, 8, 40, tzinfo=UTC),
            datetime(2026, 1, 8, 11, 45, tzinfo=UTC), **self.kwargs())
        assert got is None
        assert "not closed" in why

    def test_delay_floor_boundary_admits_a_closed_case(self):
        # Free legs end 09:00 + 90 min floor = 10:30 <= 12:00: closed.
        init, window_end = case_window(
            datetime(2026, 1, 8, 6, 0, tzinfo=UTC),
            datetime(2026, 1, 8, 9, 0, tzinfo=UTC), **self.kwargs())
        assert init == datetime(2026, 1, 8, 6, 0, tzinfo=UTC)
        assert window_end == datetime(2026, 1, 8, 7, 30, tzinfo=UTC)


class TestPlaceWindow:
    NOW = datetime(2026, 1, 8, 12, 0, tzinfo=UTC)

    def kwargs(self, **over):
        base = dict(cycles=6, cycle_seconds=900, free_legs=6,
                    now=self.NOW)
        base.update(over)
        return base

    def test_free_legs_straddle_the_peak(self):
        # a 12-hour event peaking at 02:00; the window lands so the
        # graded frames sample the peak, not initiation at 18:00
        init, window_end = place_window(
            datetime(2026, 1, 6, 18, 0, tzinfo=UTC),
            datetime(2026, 1, 7, 6, 0, tzinfo=UTC),
            datetime(2026, 1, 7, 2, 0, tzinfo=UTC), **self.kwargs())
        assert init == datetime(2026, 1, 7, 0, 0, tzinfo=UTC)
        assert window_end == datetime(2026, 1, 7, 1, 30, tzinfo=UTC)
        free_end = window_end + timedelta(seconds=6 * 900)
        assert window_end <= datetime(2026, 1, 7, 2, 0, tzinfo=UTC) \
            <= free_end

    def test_peak_at_run_start_clamps_to_earliest_init(self):
        init, _ = place_window(
            datetime(2026, 1, 6, 18, 20, tzinfo=UTC),
            datetime(2026, 1, 7, 6, 0, tzinfo=UTC),
            datetime(2026, 1, 6, 18, 30, tzinfo=UTC), **self.kwargs())
        assert init == datetime(2026, 1, 6, 19, 0, tzinfo=UTC)

    def test_peak_near_now_clamps_to_the_delay_floor(self):
        # run continues to "now"; the placed window must still close
        init, window_end = place_window(
            datetime(2026, 1, 8, 0, 0, tzinfo=UTC),
            datetime(2026, 1, 8, 12, 0, tzinfo=UTC),
            datetime(2026, 1, 8, 11, 0, tzinfo=UTC), **self.kwargs())
        last = window_end + timedelta(seconds=6 * 900)
        assert (self.NOW - last).total_seconds() >= 5400.0
        assert init.minute == 0

    def test_cycling_stays_inside_the_echo_run(self):
        run_end = datetime(2026, 1, 7, 6, 0, tzinfo=UTC)
        _, window_end = place_window(
            datetime(2026, 1, 6, 18, 0, tzinfo=UTC), run_end,
            datetime(2026, 1, 7, 5, 55, tzinfo=UTC), **self.kwargs())
        assert window_end <= run_end


class TestDedup:
    def case(self, site, lat, lon, init, window_end, rank):
        return {"site": site, "lat": lat, "lon": lon, "init": init,
                "window_end": window_end, "rank_score": rank}

    def test_neighboring_radars_same_window_collapse(self):
        cases = [
            self.case("XAAA", 41.7, -93.7, "2026-01-07T04:00:00Z",
                      "2026-01-07T05:30:00Z", 9000),
            self.case("XBBB", 42.6, -93.6, "2026-01-07T04:00:00Z",
                      "2026-01-07T05:30:00Z", 7000),
        ]
        kept, dropped = dedup_cases(cases, dedup_km=300, dedup_hours=3)
        assert [c["site"] for c in kept] == ["XAAA"]
        assert len(dropped) == 1
        assert "same event" in dropped[0]["dropped_for"]

    def test_far_radars_are_separate_events(self):
        cases = [
            self.case("XAAA", 41.7, -93.7, "2026-01-07T04:00:00Z",
                      "2026-01-07T05:30:00Z", 9000),
            self.case("XCCC", 35.3, -97.3, "2026-01-07T04:00:00Z",
                      "2026-01-07T05:30:00Z", 7000),
        ]
        kept, dropped = dedup_cases(cases, dedup_km=300, dedup_hours=3)
        assert len(kept) == 2 and not dropped

    def test_same_site_different_day_is_two_cases(self):
        cases = [
            self.case("XAAA", 41.7, -93.7, "2026-01-06T04:00:00Z",
                      "2026-01-06T05:30:00Z", 9000),
            self.case("XAAA", 41.7, -93.7, "2026-01-07T04:00:00Z",
                      "2026-01-07T05:30:00Z", 8000),
        ]
        kept, dropped = dedup_cases(cases, dedup_km=300, dedup_hours=3)
        assert len(kept) == 2 and not dropped

    def test_same_site_adjacent_windows_collapse(self):
        cases = [
            self.case("XAAA", 41.7, -93.7, "2026-01-07T04:00:00Z",
                      "2026-01-07T05:30:00Z", 9000),
            self.case("XAAA", 41.7, -93.7, "2026-01-07T06:00:00Z",
                      "2026-01-07T07:30:00Z", 8000),
        ]
        kept, _ = dedup_cases(cases, dedup_km=300, dedup_hours=3)
        assert len(kept) == 1

    def test_haversine_sanity(self):
        assert haversine_km(41.7, -93.7, 41.7, -93.7) == 0.0
        assert 90 < haversine_km(41.7, -93.7, 42.6, -93.6) < 120


class TestAggregation:
    def frames(self, fss_f, fss_c, obs=1000, fcst=900):
        return [{"lead_minutes": 15.0 * (i + 1), "fss30_fcst": f,
                 "fss30_control": c, "obs_cols_gt35": obs,
                 "fcst_cols_gt35_in_echo": fcst,
                 "control_cols_gt35_in_echo": 300}
                for i, (f, c) in enumerate(zip(fss_f, fss_c))]

    def test_median_and_iqr_not_just_mean(self):
        rows = [
            {"case_id": "a", "frames": self.frames([0.70], [0.30])},
            {"case_id": "b", "frames": self.frames([0.74], [0.32])},
            {"case_id": "c", "frames": self.frames([0.10], [0.28])},
        ]
        agg = aggregate_frames(rows)
        lead = agg["per_lead"][0]
        assert lead["fss30_fcst"]["median"] == pytest.approx(0.70)
        assert lead["fss30_fcst"]["min"] == pytest.approx(0.10)
        assert lead["fss30_fcst"]["n"] == 3
        # the outlier is visible in the spread, not smeared into a mean
        assert lead["fss30_fcst"]["mean"] < lead["fss30_fcst"]["median"]

    def test_control_relative_and_bias(self):
        rows = [{"case_id": "a",
                 "frames": self.frames([0.7, 0.6], [0.3, 0.4],
                                       obs=2800, fcst=2000)}]
        agg = aggregate_frames(rows)
        first = agg["per_lead"][0]
        assert first["fss30_delta_vs_control"]["median"] == \
            pytest.approx(0.4)
        assert first["cases_beating_control"] == 1
        # aggregate values are rounded to 4 places in the receipt
        assert first["column_count_bias"]["median"] == \
            pytest.approx(2000 / 2800, abs=5e-5)

    def test_structure_metrics_flagged_when_present(self):
        frames = self.frames([0.7], [0.3])
        frames[0]["structure"] = {"objects": {"count": 4}}
        agg = aggregate_frames([{"case_id": "a", "frames": frames}])
        assert agg["structure_metrics_present"] is True
        agg2 = aggregate_frames(
            [{"case_id": "a", "frames": self.frames([0.7], [0.3])}])
        assert agg2["structure_metrics_present"] is False

    def test_lead_minutes(self):
        assert lead_minutes("2026-01-07T05:45:00Z",
                            "2026-01-07T05:30:00Z") == 15.0


class TestFrozenConfig:
    def test_hash_is_stable_and_order_free(self):
        left = config_hash({"a": 1, "b": 2.0})
        right = config_hash({"b": 2.0, "a": 1})
        assert left == right
        assert left != config_hash({"a": 1, "b": 2.5})

    def test_baseline_shape_is_the_shipped_one(self):
        assert BASELINE_CONFIG["members"] == 10
        assert BASELINE_CONFIG["dx_km"] == 3.0
        assert BASELINE_CONFIG["cycles"] == 6
        assert BASELINE_CONFIG["free_legs"] == 6
        assert BASELINE_CONFIG["history_interval_seconds"] == 900.0

    def test_front_door_argv_is_frozen_and_delayed(self, tmp_path):
        case = {"site": "XAAA", "window_end": "2026-01-07T05:30:00Z"}
        argv = front_door_argv(case, case_dir=tmp_path / "c",
                               config=BASELINE_CONFIG,
                               polygon=tmp_path / "p.geojson")
        text = " ".join(argv)
        assert "--no-verify" in argv          # one-shot verify follows
        assert "--allow-stale" in argv        # archived window
        assert "--domain-polygon" in text     # census siting, not now's
        assert "--members 10" in text
        assert "--history-interval-seconds 900" in text
        # nothing about the case leaks into tuning: the only
        # case-specific argv entries are site, window, and paths
        assert "--seed" not in text           # front-door default

    def test_variant_keys_forward_only_when_the_config_carries_them(
            self, tmp_path):
        """An A/B arm differs by exactly its stated key(s), nothing else.

        A config frozen before the variant keys existed must keep
        producing the byte-same argv (its hash is stored); a variant
        config carries its keys through to the front door.
        """
        case = {"site": "XAAA", "window_end": "2026-01-07T05:30:00Z"}
        baseline = front_door_argv(
            case, case_dir=tmp_path / "c", config=BASELINE_CONFIG,
            polygon=tmp_path / "p.geojson")
        for flag in ("--hydrometeors", "--positivity-policy",
                     "--reflectivity-analysis"):
            assert flag not in baseline
        variant_config = dict(BASELINE_CONFIG)
        variant_config.update({
            "hydrometeors": True, "positivity_policy": "clip",
            "reflectivity_analysis": True})
        variant = front_door_argv(
            case, case_dir=tmp_path / "c", config=variant_config,
            polygon=tmp_path / "p.geojson")
        assert variant[:len(baseline)] == baseline
        assert "--hydrometeors" in variant
        assert variant[variant.index("--positivity-policy") + 1] == "clip"
        assert "--reflectivity-analysis" in variant


class TestDelayedWindowIsEnforced:
    """The equivalence claim has to be arithmetic, not prose."""

    NOW = datetime(2026, 1, 8, 12, 0, tzinfo=UTC)

    def test_a_closed_case_passes(self):
        # free legs end 09:00; 3 h old at 12:00, well past the 90 min floor
        closed, why = case_is_closed("2026-01-08T07:30:00Z", free_legs=6,
                                     cycle_seconds=900, now=self.NOW)
        assert closed and why == ""

    def test_an_unclosed_case_is_refused_with_the_arithmetic(self):
        # free legs end 11:45: 15 minutes old, floor is 90
        closed, why = case_is_closed("2026-01-08T10:15:00Z", free_legs=6,
                                     cycle_seconds=900, now=self.NOW)
        assert not closed
        assert "15 min old" in why and "90 min" in why

    def test_the_boundary_is_inclusive(self):
        last = self.NOW - timedelta(seconds=DELAY_FLOOR_SECONDS)
        window_end = last - timedelta(seconds=6 * 900)
        closed, _ = case_is_closed(
            window_end.strftime("%Y-%m-%dT%H:%M:%SZ"), free_legs=6,
            cycle_seconds=900, now=self.NOW)
        assert closed

    def test_the_floor_is_not_the_plans_to_choose(self):
        # The runner passes no delay_floor_seconds, so a plan carrying a
        # smaller one cannot talk its way past this predicate.
        closed, _ = case_is_closed("2026-01-08T10:15:00Z", free_legs=6,
                                   cycle_seconds=900, now=self.NOW)
        assert not closed
        relaxed, _ = case_is_closed("2026-01-08T10:15:00Z", free_legs=6,
                                    cycle_seconds=900, now=self.NOW,
                                    delay_floor_seconds=60.0)
        assert relaxed          # only an explicit argument can relax it


class TestCoverageCapsAreRecorded:
    def cases(self, n):
        return [{"site": f"X{i:03d}", "window_end": "2026-01-07T05:30:00Z",
                 "census_median_gates": 9000 - 100 * i} for i in range(n)]

    def test_under_the_cap_drops_nothing(self):
        kept, dropped = apply_max_cases(self.cases(4), 10)
        assert len(kept) == 4 and dropped == []

    def test_the_cap_hands_back_what_it_removed(self):
        kept, dropped = apply_max_cases(self.cases(14), 10)
        assert len(kept) == 10
        assert len(dropped) == 4
        # the dropped ones are the SMALLEST storms, and say so
        assert [c["census_median_gates"] for c in dropped] == \
            [8000, 7900, 7800, 7700]
        assert "over --max-cases 10" in dropped[0]["dropped_for"]
        assert "ranked 11 of 14" in dropped[0]["dropped_for"]

    def test_the_kept_cases_are_untouched_copies(self):
        source = self.cases(12)
        kept, dropped = apply_max_cases(source, 10)
        assert "dropped_for" not in kept[0]
        assert "dropped_for" not in source[10]   # original not mutated


class TestCampaignVerdict:
    def test_complete_only_when_every_planned_case_graded(self):
        line = campaign_verdict(graded=10, planned=10, failed=[],
                                refused=[], unreached=[])
        assert line.startswith("CAMPAIGN_COMPLETE 10/10")

    def test_cases_the_card_never_freed_for_stay_in_the_denominator(self):
        # the regression this exists for: three of twelve cases ran, the
        # card stayed held, and the verdict read "COMPLETE 3/4"
        line = campaign_verdict(graded=3, planned=12, failed=[],
                                refused=[],
                                unreached=["c-3", "c-4", "c-5"])
        assert line.startswith("CAMPAIGN_INCOMPLETE 3/12")
        assert "never run: c-3, c-4, c-5" in line

    def test_failed_and_refused_are_different_words(self):
        line = campaign_verdict(graded=1, planned=3, failed=["a"],
                                refused=["b"], unreached=[])
        assert "failed: a" in line and "refused: b" in line
        assert line.startswith("CAMPAIGN_INCOMPLETE 1/3")

    def test_a_full_plan_with_a_failure_is_not_complete(self):
        line = campaign_verdict(graded=9, planned=10, failed=["x"],
                                refused=[], unreached=[])
        assert line.startswith("CAMPAIGN_INCOMPLETE")


class TestSkipReason:
    def test_a_recorded_failure_is_quoted(self, tmp_path):
        arms = tmp_path / "arms"
        arms.mkdir()
        (arms / "c-1.failed").write_text("FAILED (verify exit 3) after "
                                         "88.0 s\n", encoding="utf-8")
        why = skip_reason(tmp_path, "c-1")
        assert "recorded a failure" in why and "verify exit 3" in why

    def test_a_case_that_ran_but_graded_nothing(self, tmp_path):
        case = tmp_path / "cases" / "c-2"
        case.mkdir(parents=True)
        (case / "nowcast-receipt.json").write_text("{}", encoding="utf-8")
        assert "graded no frame" in skip_reason(tmp_path, "c-2")

    def test_a_case_never_reached(self, tmp_path):
        assert "never started" in skip_reason(tmp_path, "c-3")

    def test_the_three_reasons_are_distinguishable(self, tmp_path):
        arms = tmp_path / "arms"
        arms.mkdir()
        (arms / "a.failed").write_text("FAILED (frontdoor exit 2)\n",
                                       encoding="utf-8")
        (tmp_path / "cases" / "b").mkdir(parents=True)
        (tmp_path / "cases" / "b" / "nowcast-receipt.json").write_text(
            "{}", encoding="utf-8")
        reasons = {skip_reason(tmp_path, name) for name in ("a", "b", "c")}
        assert len(reasons) == 3
