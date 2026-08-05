"""The draw-a-box launcher: geometry, cost, refusals, and the page.

Pure tests: no network, no GPU, no browser.  The wizard subprocess is
stubbed where a test is about what the launcher does with its answer
rather than about the answer.  Sites here are synthetic ids -- real
station names never enter the tree's generic code or its fixtures
(standing owner rule).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.da_nowcast_launcher import (
    CYCLE_BUDGET_REFUSE_FACTOR, MIN_CELLS_PER_SIDE, REFERENCE, SCHEMA,
    VOLUME_INTERVAL_SECONDS, LauncherError, box_span_km, build_page,
    build_parser, cost_estimate, cycle_budget_verdict, decimate,
    geojson_from_box, great_circle_km, launch_argv, normalize_box,
    plan_box, run_name, safe_run_dir, site_coverage, size_verdict,
    trajectory_leg_seconds, wizard_cmd)


def sites():
    """A synthetic site table shaped like the vendored one."""

    return [
        {"id": "AAAA", "name": "one", "lat_deg": 35.0, "lon_deg": -97.0},
        {"id": "BBBB", "name": "two", "lat_deg": 35.0, "lon_deg": -95.0},
        {"id": "CCCC", "name": "far", "lat_deg": 45.0, "lon_deg": -80.0},
    ]


BOX = normalize_box(-98.0, 34.0, -96.0, 36.0)


# ---------------------------------------------------------------------------
# the box a caller drags
# ---------------------------------------------------------------------------
class TestNormalizeBox:
    def test_drag_direction_is_not_information(self):
        a = normalize_box(-96.0, 36.0, -98.0, 34.0)
        b = normalize_box(-98.0, 34.0, -96.0, 36.0)
        assert a == b

    def test_centre_is_the_middle(self):
        assert BOX["center_lat"] == 35.0
        assert BOX["center_lon"] == -97.0

    def test_a_line_is_not_a_box(self):
        with pytest.raises(LauncherError, match="not a box"):
            normalize_box(-98.0, 34.0, -98.0, 36.0)
        with pytest.raises(LauncherError, match="not a box"):
            normalize_box(-98.0, 34.0, -96.0, 34.0)

    def test_the_poles_are_refused(self):
        with pytest.raises(LauncherError, match="between the poles"):
            normalize_box(-98.0, 80.0, -96.0, 91.0)

    def test_off_the_earth_is_refused(self):
        with pytest.raises(LauncherError, match="not a box"):
            normalize_box(-190.0, 34.0, -96.0, 36.0)

    def test_span_shrinks_with_latitude(self):
        low = box_span_km(normalize_box(-98.0, 0.0, -96.0, 2.0))
        high = box_span_km(normalize_box(-98.0, 60.0, -96.0, 62.0))
        assert low[0] > high[0]
        assert low[1] == pytest.approx(high[1])

    def test_geojson_ring_closes(self):
        ring = geojson_from_box(BOX)["coordinates"][0]
        assert ring[0] == ring[-1]
        assert len(ring) == 5


# ---------------------------------------------------------------------------
# which radar can see it
# ---------------------------------------------------------------------------
class TestSiteCoverage:
    def test_a_site_inside_the_box_covers_it_fully(self):
        found = site_coverage(sites(), BOX, range_km=250.0)
        assert found[0]["id"] == "AAAA"
        assert found[0]["coverage"] == "full"

    def test_a_distant_site_is_not_offered_at_all(self):
        assert "CCCC" not in [s["id"] for s in
                              site_coverage(sites(), BOX,
                                            range_km=250.0)]

    def test_partial_coverage_is_labelled_not_hidden(self):
        found = site_coverage(sites(), BOX, range_km=200.0)
        by_id = {s["id"]: s for s in found}
        assert by_id["AAAA"]["coverage"] == "full"
        assert by_id["BBBB"]["coverage"] == "partial"

    def test_full_coverage_sorts_ahead_of_partial(self):
        found = site_coverage(sites(), BOX, range_km=200.0)
        assert [s["coverage"] for s in found] == ["full", "partial"]

    def test_nothing_in_range_is_an_empty_list(self):
        far = normalize_box(10.0, 10.0, 11.0, 11.0)
        assert site_coverage(sites(), far, range_km=250.0) == []

    def test_distance_is_great_circle_not_flat(self):
        # one degree of latitude, anywhere
        assert great_circle_km(35.0, -97.0, 36.0, -97.0) == \
            pytest.approx(111.2, abs=0.5)
        assert great_circle_km(35.0, -97.0, 35.0, -97.0) == 0.0


# ---------------------------------------------------------------------------
# what it will cost
# ---------------------------------------------------------------------------
class TestCost:
    def test_the_reference_reproduces_itself(self):
        seconds = trajectory_leg_seconds(
            nx=REFERENCE["nx"], ny=REFERENCE["ny"], nz=REFERENCE["nz"],
            dt_s=REFERENCE["dt_s"],
            leg_seconds=REFERENCE["leg_seconds"])
        assert seconds == pytest.approx(
            REFERENCE["seconds_per_trajectory_leg"])

    def test_twice_the_cells_is_twice_the_work(self):
        one = trajectory_leg_seconds(nx=100, ny=100, nz=40, dt_s=15.0,
                                     leg_seconds=900.0)
        two = trajectory_leg_seconds(nx=200, ny=100, nz=40, dt_s=15.0,
                                     leg_seconds=900.0)
        assert two == pytest.approx(2 * one)

    def test_twice_the_steps_is_twice_the_work(self):
        one = trajectory_leg_seconds(nx=100, ny=100, nz=40, dt_s=15.0,
                                     leg_seconds=900.0)
        two = trajectory_leg_seconds(nx=100, ny=100, nz=40, dt_s=15.0,
                                     leg_seconds=1800.0)
        assert two == pytest.approx(2 * one)

    def test_a_shorter_timestep_costs_more_for_the_same_leg(self):
        coarse = trajectory_leg_seconds(nx=100, ny=100, nz=40,
                                        dt_s=15.0, leg_seconds=900.0)
        fine = trajectory_leg_seconds(nx=100, ny=100, nz=40, dt_s=5.0,
                                      leg_seconds=900.0)
        assert fine == pytest.approx(3 * coarse)

    def test_the_control_trajectory_is_counted(self):
        cost = cost_estimate(nx=132, ny=132, nz=49, dt_s=15.0,
                             members=10, cycle_seconds=300.0,
                             free_legs=6, free_leg_seconds=900.0)
        assert cost["trajectories"] == 11

    def test_cost_is_linear_in_the_ensemble_size(self):
        def total(members):
            return cost_estimate(
                nx=132, ny=132, nz=49, dt_s=15.0, members=members,
                cycle_seconds=300.0, free_legs=6,
                free_leg_seconds=900.0)["cycle_seconds_total"]

        small, large = total(10), total(36)
        overhead = REFERENCE["process_overhead_seconds"]
        assert (large - overhead) == pytest.approx(
            (small - overhead) * 37 / 11, rel=0.05)

    def test_no_free_legs_means_no_refresh_cost(self):
        cost = cost_estimate(nx=132, ny=132, nz=49, dt_s=15.0,
                             members=10, cycle_seconds=300.0,
                             free_legs=0, free_leg_seconds=900.0)
        assert cost["forecast_refresh_seconds"] == 0.0
        assert cost["free_forecast_minutes"] == 0

    def test_the_basis_of_the_estimate_travels_with_it(self):
        cost = cost_estimate(nx=132, ny=132, nz=49, dt_s=15.0,
                             members=10, cycle_seconds=300.0,
                             free_legs=6, free_leg_seconds=900.0)
        assert cost["basis"]["device"]
        assert "measurement" in cost["basis"]["caveat"] \
            or "measured" in cost["basis"]["caveat"]

    def test_the_measured_run_is_reproduced_within_a_quarter(self):
        # 6 applied 900 s cycles + 6 free 900 s legs at N=10 on
        # 132x132x49 measured 506 s end to end on 2026-08-05.
        cost = cost_estimate(nx=132, ny=132, nz=49, dt_s=15.0,
                             members=10, cycle_seconds=900.0,
                             free_legs=6, free_leg_seconds=900.0)
        modelled = (6 * (cost["assimilation_seconds"]
                         - REFERENCE["process_overhead_seconds"])
                    + cost["forecast_refresh_seconds"]
                    + REFERENCE["process_overhead_seconds"])
        assert modelled == pytest.approx(506.0, rel=0.25)


# ---------------------------------------------------------------------------
# refusing a box, politely and specifically
# ---------------------------------------------------------------------------
class TestVerdicts:
    def test_a_cycle_inside_the_volume_interval_is_fine(self):
        verdict = cycle_budget_verdict(120.0)
        assert verdict["level"] == "ok"

    def test_slower_than_the_feed_is_a_warning_not_a_refusal(self):
        verdict = cycle_budget_verdict(VOLUME_INTERVAL_SECONDS * 2)
        assert verdict["level"] == "warn"
        assert "behind" in verdict["message"]

    def test_hopelessly_slower_is_refused_with_the_ratio(self):
        verdict = cycle_budget_verdict(
            VOLUME_INTERVAL_SECONDS * (CYCLE_BUDGET_REFUSE_FACTOR + 1))
        assert verdict["level"] == "refuse"
        assert "never catch up" in verdict["message"]

    def test_a_big_enough_grid_has_no_size_verdict(self):
        assert size_verdict(120, 120, 3.0) is None

    def test_a_grid_too_small_for_a_storm_is_refused(self):
        verdict = size_verdict(40, 120, 3.0)
        assert verdict["level"] == "refuse"
        assert str(MIN_CELLS_PER_SIDE) in verdict["message"]
        assert "leave through the boundary" in verdict["message"]

    def test_the_short_side_is_what_counts(self):
        assert size_verdict(500, 20, 3.0) is not None


# ---------------------------------------------------------------------------
# the plan a caller is shown, composed from every source
# ---------------------------------------------------------------------------
def stub_fit(monkeypatch, **fields):
    import tools.da_nowcast_launcher as mod

    payload = {"ok": True, "argv": ["gpuwm", "domain"], "nx": 124,
               "ny": 114, "nz": 49, "dx_km": 3.0, "dt_s": 15.0,
               "polygon": "box.geojson", "explain_tail": ["sizing"]}
    payload.update(fields)
    monkeypatch.setattr(mod, "fit_box",
                        lambda *a, **k: dict(payload))


class TestPlanBox:
    def plan(self, **overrides):
        kwargs = dict(dx_km=3.0, members=10, free_legs=6,
                      free_leg_seconds=900.0, cycle_seconds=300.0,
                      profile="a-profile-v1", epoch_hours=4,
                      range_km=250.0, vram_gib=32.0,
                      work_dir=Path("."), sites=sites())
        kwargs.update(overrides)
        return plan_box(BOX, **kwargs)

    def test_a_good_box_is_a_go_with_a_grid_a_cost_and_sites(
            self, monkeypatch):
        stub_fit(monkeypatch)
        plan = self.plan()
        assert plan["schema"] == SCHEMA
        assert plan["ok"] is True
        assert plan["grid"]["nx"] == 124
        assert plan["cost"]["cycle_seconds_total"] > 0
        assert plan["sites"][0]["id"] == "AAAA"

    def test_the_wizards_refusal_is_passed_through_untouched(
            self, monkeypatch):
        stub_fit(monkeypatch, ok=False,
                 refusal="gpuwm check: FAIL, budget exceeded")
        plan = self.plan()
        assert plan["ok"] is False
        assert plan["grid"] is None
        assert plan["refusals"][0]["source"] == "gpuwm domain"
        assert "budget exceeded" in plan["refusals"][0]["message"]

    def test_a_tiny_box_is_refused_even_when_the_wizard_is_happy(
            self, monkeypatch):
        stub_fit(monkeypatch, nx=30, ny=30)
        plan = self.plan()
        assert plan["ok"] is False
        assert any(r["source"] == "domain size"
                   for r in plan["refusals"])

    def test_an_unaffordable_ensemble_is_refused_on_the_clock(
            self, monkeypatch):
        stub_fit(monkeypatch)
        plan = self.plan(members=96)
        assert plan["ok"] is False
        assert any(r["source"] == "cycle budget"
                   for r in plan["refusals"])

    def test_no_radar_in_range_is_a_refusal(self, monkeypatch):
        stub_fit(monkeypatch)
        ocean = normalize_box(-30.0, 10.0, -28.0, 12.0)
        plan = plan_box(
            ocean, dx_km=3.0, members=10, free_legs=6,
            free_leg_seconds=900.0, cycle_seconds=300.0,
            profile="a-profile-v1", epoch_hours=4, range_km=250.0,
            vram_gib=32.0, work_dir=Path("."), sites=sites())
        assert plan["ok"] is False
        assert any(r["source"] == "radar coverage"
                   for r in plan["refusals"])
        assert "no observations to assimilate" in             plan["refusals"][0]["message"]

    def test_partial_coverage_only_warns(self, monkeypatch):
        stub_fit(monkeypatch)
        plan = self.plan(range_km=100.0)
        assert plan["ok"] is True
        assert any(w["source"] == "radar coverage"
                   for w in plan["warnings"])

    def test_the_requested_settings_are_echoed_back(self, monkeypatch):
        stub_fit(monkeypatch)
        plan = self.plan(members=24, free_legs=8)
        assert plan["requested"]["members"] == 24
        assert plan["requested"]["free_legs"] == 8

    def test_the_plan_is_json(self, monkeypatch):
        stub_fit(monkeypatch)
        json.dumps(self.plan(), default=str)


# ---------------------------------------------------------------------------
# the command the page actually runs
# ---------------------------------------------------------------------------
class TestLaunchArgv:
    def argv(self, **overrides):
        requested = {"members": 24, "free_legs": 8,
                     "free_leg_seconds": 900.0, "dx_km": 3.0,
                     "physics_profile": "a-profile-v1",
                     "epoch_hours": 4}
        requested.update(overrides)
        return launch_argv(site="QQQQ", out=Path("out"),
                           polygon=Path("box.geojson"),
                           plan={"requested": requested},
                           run_root=Path("run"))

    def test_it_starts_the_daemon_not_a_one_shot(self):
        argv = self.argv()
        assert "tools.da_nowcast_auto" in argv
        assert "start" in argv

    def test_the_drawn_box_is_the_domain(self):
        argv = self.argv()
        assert argv[argv.index("--polygon") + 1] == "box.geojson"

    def test_ensemble_size_reaches_the_daemon(self):
        argv = self.argv(members=36)
        assert argv[argv.index("--members") + 1] == "36"

    def test_the_run_root_is_stated_not_inferred(self):
        assert "--run-root" in self.argv()

    def test_the_site_is_an_argument(self):
        argv = self.argv()
        assert argv[argv.index("--site") + 1] == "QQQQ"


# ---------------------------------------------------------------------------
# a browser cannot climb out of the work root
# ---------------------------------------------------------------------------
class TestRunNames:
    def test_a_generated_name_is_accepted(self, tmp_path):
        assert safe_run_dir(tmp_path, run_name()).parent == \
            tmp_path.resolve()

    @pytest.mark.parametrize("name", [
        "..", "../escape", "a/b", "a\\b", "", "/etc", "C:", "." * 3,
        "x" * 65])
    def test_anything_that_could_escape_is_refused(self, tmp_path,
                                                   name):
        with pytest.raises(LauncherError, match="not a run name"):
            safe_run_dir(tmp_path, name)


# ---------------------------------------------------------------------------
# the page loads with the network unplugged
# ---------------------------------------------------------------------------
class TestPage:
    def page(self):
        return build_page(
            basemap={"coast": [[[-100.0, 40.0], [-99.0, 41.0]]],
                     "nation": [], "state": []},
            sites=[{"id": "AAAA", "lat_deg": 35.0, "lon_deg": -97.0}],
            extent=(-126.0, -66.0, 23.0, 50.5),
            defaults={"dx_km": 3.0, "members": 10, "free_legs": 6,
                      "free_leg_seconds": 900.0})

    def test_nothing_is_fetched_from_the_network_to_draw_it(self):
        page = self.page()
        for forbidden in ("http://", "https://", "//cdn", "@import",
                          "src=\"http", "integrity="):
            assert forbidden not in page, forbidden

    def test_no_external_script_or_stylesheet_is_referenced(self):
        page = self.page()
        assert "<script src" not in page
        assert "<link" not in page

    def test_the_basemap_and_sites_are_inlined(self):
        page = self.page()
        assert "AAAA" in page
        assert "-99" in page

    def test_the_defaults_reach_the_controls(self):
        assert '"members":10' in self.page().replace(", ", ",")

    def test_it_says_what_it_is(self):
        page = self.page()
        assert "UNSCORED" in page
        assert "Nothing starts until you press Start" in page

    def test_decimation_keeps_the_endpoints(self):
        pts = [[float(i), 0.0] for i in range(10)]
        kept = decimate(pts, step=4)
        assert kept[0] == [0.0, 0.0]
        assert kept[-1] == [9.0, 0.0]

    def test_decimation_of_a_short_segment_is_a_copy(self):
        assert decimate([[0.0, 0.0], [1.0, 1.0]], step=8) == \
            [[0.0, 0.0], [1.0, 1.0]]


# ---------------------------------------------------------------------------
# the wizard is driven, not reimplemented
# ---------------------------------------------------------------------------
class TestWizardCommand:
    def test_it_asks_the_shipped_wizard(self):
        argv = wizard_cmd(polygon=Path("b.geojson"),
                          out_toml=Path("p.toml"), dx_km=3.0,
                          profile="a-profile-v1", hours=4,
                          cycle="2026-08-05T00", name="plan",
                          vram_gib=32.0)
        assert argv[1:4] == ["-m", "gpuwm.cli", "domain"]
        assert "--explain" in argv
        assert argv[argv.index("--vram-gib") + 1] == "32"

    def test_no_card_named_means_no_flag(self):
        argv = wizard_cmd(polygon=Path("b"), out_toml=Path("p"),
                          dx_km=3.0, profile="p", hours=4,
                          cycle="c", name="n", vram_gib=None)
        assert "--vram-gib" not in argv


# ---------------------------------------------------------------------------
# argument surface
# ---------------------------------------------------------------------------
class TestArguments:
    def test_serve_binds_locally_by_default(self):
        args = build_parser().parse_args(
            ["serve", "--work-root", "w"])
        assert args.host == "127.0.0.1"

    def test_ensemble_size_is_a_first_class_flag_everywhere(self):
        for mode, extra in (("serve", ["--work-root", "w"]),
                            ("plan", ["--box=-1,1,2,3"]),
                            ("page", ["--out", "p.html"])):
            args = build_parser().parse_args(
                [mode, *extra, "--members", "36"])
            assert args.members == 36

    def test_defaults_match_the_daemons(self):
        args = build_parser().parse_args(["page", "--out", "p.html"])
        assert (args.members, args.free_legs,
                args.free_leg_seconds) == (10, 6, 900.0)


# ---------------------------------------------------------------------------
# the routes the page drives, over a real socket
# ---------------------------------------------------------------------------
class TestServer:
    """The page is a thin driver, so what it drives has to answer.

    A real ThreadingHTTPServer on an ephemeral port, shut down in
    process when the test is done -- the seam is HTTP, so testing it
    over anything else would be testing a different thing.  The wizard
    and the site table are stubbed: this is about the routes.
    """

    def server(self, tmp_path, monkeypatch):
        import threading

        import tools.da_nowcast_launcher as mod
        from http.server import ThreadingHTTPServer

        monkeypatch.setattr(mod, "read_site_table", lambda: sites())
        monkeypatch.setattr(
            mod, "basemap_polylines",
            lambda extent, **k: {"coast": [], "nation": [], "state": []})
        stub_fit(monkeypatch)
        state = mod.LauncherState(
            work_root=tmp_path / "work", run_root=tmp_path,
            defaults={"dx_km": 3.0, "members": 10, "free_legs": 6,
                      "free_leg_seconds": 900.0,
                      "cycle_seconds": 300.0,
                      "physics_profile": "a-profile-v1",
                      "epoch_hours": 4},
            extent=(-126.0, -66.0, 23.0, 50.5), vram_gib=32.0,
            range_km=250.0)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                                    mod.make_handler(state))
        thread = threading.Thread(target=httpd.serve_forever,
                                  daemon=True)
        thread.start()
        return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"

    def get(self, base, path):
        from urllib.request import urlopen
        from urllib.error import HTTPError

        try:
            with urlopen(base + path) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def post(self, base, path, payload):
        import json as _json
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError

        request = Request(
            base + path, data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urlopen(request) as response:
                return response.status, _json.loads(response.read())
        except HTTPError as error:
            return error.code, _json.loads(error.read())

    def test_the_routes_answer(self, tmp_path, monkeypatch):
        httpd, base = self.server(tmp_path, monkeypatch)
        try:
            code, body = self.get(base, "/")
            assert code == 200 and b"draw a box" in body

            code, body = self.get(base, "/api/sites")
            assert code == 200
            assert len(json.loads(body)["sites"]) == 3

            code, plan = self.post(base, "/api/plan", {
                "box": {"west": -98.0, "south": 34.0, "east": -96.0,
                        "north": 36.0},
                "dx_km": 3.0, "members": 10, "free_legs": 6,
                "free_leg_seconds": 900.0})
            assert code == 200 and plan["ok"] is True
            assert plan["grid"]["nx"] == 124
            assert plan["sites"][0]["id"] == "AAAA"

            # a box that is not one is refused, not crashed on
            code, bad = self.post(base, "/api/plan", {
                "box": {"west": -98.0, "south": 34.0, "east": -98.0,
                        "north": 36.0},
                "dx_km": 3.0, "members": 10, "free_legs": 6,
                "free_leg_seconds": 900.0})
            assert code == 400 and bad["ok"] is False

            # no run yet, so the status route says so rather than 500s
            code, _ = self.get(base, "/api/status?run=nowcast-x")
            assert code == 404

            # and a browser cannot walk out of the work root
            code, _ = self.get(base, "/api/status?run=..")
            assert code == 400
            code, _ = self.get(base, "/runs/../gallery/index.html")
            assert code in (400, 404)
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestCanvasProportions:
    """A box drawn on a stretched map is a box the caller did not mean."""

    def test_the_canvas_matches_the_ground_it_shows(self):
        from tools.da_nowcast_launcher import DEFAULT_EXTENT, canvas_size

        width, height = canvas_size(DEFAULT_EXTENT)
        west, east, south, north = DEFAULT_EXTENT
        import math
        mid = math.radians((south + north) / 2.0)
        ground = ((east - west) * math.cos(mid)) / (north - south)
        assert width / height == pytest.approx(ground, rel=0.01)

    def test_a_square_of_ground_is_a_square_of_pixels(self):
        from tools.da_nowcast_launcher import canvas_size

        # one degree of latitude, and the longitude span that matches it
        import math
        south, north = 34.0, 35.0
        span = 1.0 / math.cos(math.radians(34.5))
        width, height = canvas_size((-97.0, -97.0 + span, south, north),
                                    width=600)
        assert height == pytest.approx(600, abs=2)

    def test_a_degenerate_extent_is_refused(self):
        from tools.da_nowcast_launcher import canvas_size

        with pytest.raises(LauncherError, match="not a map extent"):
            canvas_size((-97.0, -97.0, 34.0, 35.0))

    def test_the_page_carries_the_computed_canvas(self):
        from tools.da_nowcast_launcher import DEFAULT_EXTENT, canvas_size

        width, height = canvas_size(DEFAULT_EXTENT)
        page = build_page(
            basemap={"coast": [], "nation": [], "state": []},
            sites=[], extent=DEFAULT_EXTENT,
            defaults={"dx_km": 3.0, "members": 10, "free_legs": 6,
                      "free_leg_seconds": 900.0})
        assert f'width="{width}" height="{height}"' in page


class TestCardCarriesThrough:
    def test_the_planned_card_is_the_launched_card(self):
        argv = launch_argv(
            site="QQQQ", out=Path("out"), polygon=Path("box.geojson"),
            plan={"requested": {"members": 10, "free_legs": 6,
                                "free_leg_seconds": 900.0, "dx_km": 3.0,
                                "physics_profile": "p", "epoch_hours": 4}},
            run_root=Path("run"), vram_gib=31.84)
        assert argv[argv.index("--vram-gib") + 1] == "31.84"

    def test_no_card_detected_means_no_flag_invented(self):
        argv = launch_argv(
            site="QQQQ", out=Path("out"), polygon=Path("box.geojson"),
            plan={"requested": {"members": 10, "free_legs": 6,
                                "free_leg_seconds": 900.0, "dx_km": 3.0,
                                "physics_profile": "p", "epoch_hours": 4}},
            run_root=Path("run"), vram_gib=None)
        assert "--vram-gib" not in argv
