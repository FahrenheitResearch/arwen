"""The nowcast heating evaluation's case and arm tables, overlays and window classes.

CPU only.  Covers the refusals tools/heat_eval names: an arm that changes a
second knob, a window set that does not divide the forced period, an oracle
window in a forecast arm, and a missing window; and the valid-time keying
an arm that starts an hour early needs.
"""

from __future__ import annotations

import copy
import json
import math
import tomllib
from datetime import timedelta
from pathlib import Path

import pytest

from tools.heat_eval import make_arm_configs as mac
from tools.heat_eval import plan as planmod
from tools.heat_eval.plan import PlanError


@pytest.fixture(scope="module")
def table():
    return planmod.load_arms()


@pytest.fixture(scope="module")
def anchor():
    return planmod.case_by_id("c5-20261001")


def test_the_shipped_tables_load_in_run_order(table):
    cases = planmod.load_cases()
    assert [c.id for c in cases][:5] == [
        "c5-20261001", "c1-20240521", "c4-20260822", "c2-20250314", "c3-20260802"]
    assert [c.role for c in cases].count("decision") == 4
    assert {c.role for c in cases} == {"anchor", "decision", "reserve"}
    assert [a.id for a in table.arms] == ["A", "B", "C", "E", "C60", "E3D"]
    decision = planmod.case_by_id("c1-20240521")
    anchor = planmod.case_by_id("c5-20261001")
    assert [a.id for a in table.for_case(decision, "core")] == ["A", "B", "C", "E"]
    assert [a.id for a in table.for_case(decision, "full")] == ["A", "B", "C", "E", "C60", "E3D"]
    # The optional arms run on the four decision cases only.
    assert [a.id for a in table.for_case(anchor, "full")] == ["A", "B", "C", "E"]


def test_arm_starts_lengths_and_window_ends(table, anchor):
    a, b, c = table.arm("A"), table.arm("B"), table.arm("C")
    assert planmod.iso(a.start(anchor)) == "2026-10-01T18:00:00Z"
    assert planmod.iso(b.start(anchor)) == "2026-10-01T17:00:00Z"
    assert b.run_seconds("core") == 7 * 3600 and b.run_seconds("full") == 19 * 3600
    assert [planmod.stamp(t) for t in b.window_ends(anchor)] == [
        "20261001T1715Z", "20261001T1730Z", "20261001T1745Z", "20261001T1800Z"]
    ends = c.window_ends(anchor)
    assert len(ends) == 12 and planmod.stamp(ends[0]) == "20261001T1810Z"
    assert planmod.stamp(ends[-1]) == "20261001T2000Z"
    assert a.window_ends(anchor) == []


def _row(**changes):
    row = {"id": "X", "label": "x", "column": "C", "plan": "core", "run_hours": 18,
           "lead_class": "forecast",
           "heating": {"windows": "w", "window_minutes": 10, "active_minutes": 120}}
    row.update(changes)
    return row


def test_an_arm_row_that_turns_a_second_knob_is_refused():
    planmod.arm_from_row(_row())
    with pytest.raises(PlanError, match="comparison meaningless"):
        planmod.arm_from_row(_row(mp_physics=8))
    with pytest.raises(PlanError, match="not \\[radar_heating\\] keys"):
        planmod.arm_from_row(_row(heating={"windows": "w", "window_minutes": 10,
                                           "active_minutes": 120, "cu_physics": 1}))
    with pytest.raises(PlanError, match="does not divide"):
        planmod.arm_from_row(_row(heating={"windows": "w", "window_minutes": 25,
                                           "active_minutes": 120}))
    with pytest.raises(PlanError, match="future\\s+observations would be scored"):
        planmod.arm_from_row(_row(lead_class="oracle"))
    with pytest.raises(PlanError, match="no heating"):
        planmod.arm_from_row(_row(heating={}))


def test_written_configs_differ_from_arm_a_only_in_the_overlay(tmp_path, table, anchor):
    template = mac.TEMPLATE.read_text(encoding="utf-8")
    written = mac.write_case(tmp_path, anchor, table, "full", template)
    a_text = (tmp_path / anchor.id / "arms" / "A" / "experiment.toml").read_text(encoding="utf-8")
    a_doc = tomllib.loads(a_text)
    assert str(a_doc["experiment"]["start_time"]) == "2026-10-01 18:00:00"
    assert a_doc["experiment"]["run_seconds"] == 64800.0
    assert a_doc["fetch"]["cycle"] == "2026-10-01T15" and a_doc["fetch"]["hours"] == 21
    assert "radar_heating" not in a_doc
    for arm_id in ("B", "C", "E"):
        text = (tmp_path / anchor.id / "arms" / arm_id / "experiment.toml").read_text(encoding="utf-8")
        changed = mac.check_overlay(a_text, text)
        assert {key[0] for key in changed} <= {"experiment", "radar_heating"}
    b = tomllib.loads((tmp_path / anchor.id / "arms" / "B" / "experiment.toml").read_text(encoding="utf-8"))
    assert str(b["experiment"]["start_time"]) == "2026-10-01 17:00:00"
    assert b["radar_heating"]["window_minutes"] == 15 and b["radar_heating"]["active_minutes"] == 60
    assert b["radar_heating"]["windows"].endswith("c5-20261001/windows/levelii-b")
    record = json.loads((tmp_path / anchor.id / "arms" / "C" / "arm.json").read_text(encoding="utf-8"))
    assert record["lead_class"] == "forecast" and len(record["window_ends"]) == 12
    assert Path(record["prep"]).parts[-3:] == (anchor.id, "preps", "1800z-18h")
    assert record["needs"] == ["prep-1800z-18h", "windows-nowcast"]
    # One preparation per start and length (the anchor runs no C60 or E3D),
    # plus gate G0's one-hour door case.
    assert {key: row["arms"] for key, row in written["preps"].items()} == {
        "1800z-18h": ["A", "C", "E"], "1700z-19h": ["B"], "1800z-1h": ["G0"]}
    assert written["preps"]["1700z-19h"]["boundary_leads"][0] == 2
    # Each preparation holds exactly the bytes the door checks: the arm's
    # config with its [radar_heating] table cut out.
    for arm_id in ("A", "B", "C", "E"):
        arm_text = (tmp_path / anchor.id / "arms" / arm_id / "experiment.toml").read_text(encoding="utf-8")
        key = json.loads((tmp_path / anchor.id / "arms" / arm_id / "arm.json").read_text())["prep_key"]
        prep_text = (tmp_path / anchor.id / "preps" / key / "experiment.toml").read_text(encoding="utf-8")
        assert mac.table_free(arm_text) == prep_text
    assert mac.table_free(a_text) == a_text
    door = tomllib.loads((tmp_path / anchor.id / "preps" / "1800z-1h" / "experiment.toml").read_text())
    assert door["experiment"]["run_seconds"] == 3600.0 and "radar_heating" not in door


def test_a_six_hour_arm_gets_its_own_preparation(tmp_path, table):
    # The door runs the config's own run_seconds (--run-seconds only warns),
    # so C60 and E3D cannot ride the 18 h preparation.
    case = planmod.case_by_id("c1-20240521")
    written = mac.write_case(tmp_path, case, table, "full", mac.TEMPLATE.read_text(encoding="utf-8"))
    groups = {key: row["arms"] for key, row in written["preps"].items()}
    assert groups["1800z-6h"] == ["C60", "E3D"] and groups["1800z-18h"] == ["A", "C", "E"]
    c60 = (tmp_path / case.id / "arms" / "C60" / "experiment.toml").read_text(encoding="utf-8")
    assert tomllib.loads(c60)["experiment"]["run_seconds"] == 21600.0
    prep = (tmp_path / case.id / "preps" / "1800z-6h" / "experiment.toml").read_text(encoding="utf-8")
    assert mac.table_free(c60) == prep


def test_an_overlay_that_changes_any_other_key_is_refused(tmp_path, table, anchor):
    template = mac.TEMPLATE.read_text(encoding="utf-8")
    mac.write_case(tmp_path, anchor, table, "core", template)
    a_text = (tmp_path / anchor.id / "arms" / "A" / "experiment.toml").read_text(encoding="utf-8")
    c_text = (tmp_path / anchor.id / "arms" / "C" / "experiment.toml").read_text(encoding="utf-8")
    physics = c_text.replace("mp_physics = 28", "mp_physics = 8")
    assert physics != c_text
    with pytest.raises(mac.OverlayRefused, match="shared.mp_physics"):
        mac.check_overlay(a_text, physics)
    grid = c_text.replace("dx = 3000.0", "dx = 3001.0")
    with pytest.raises(mac.OverlayRefused, match="domain.\\[0\\].dx"):
        mac.check_overlay(a_text, grid)
    boundary = c_text.replace('cycle = "2026-10-01T15"', 'cycle = "2026-10-01T12"')
    with pytest.raises(mac.OverlayRefused, match="fetch.cycle"):
        mac.check_overlay(a_text, boundary)
    with pytest.raises(mac.OverlayRefused, match="unknown keys"):
        mac.check_overlay(a_text, c_text + "cu_physics = 1\n")
    with pytest.raises(mac.OverlayRefused, match="unheated control"):
        mac.check_overlay(c_text, c_text)


def _window(root: Path, end: str, *, lead_class: str | None = None) -> None:
    folder = root / planmod.stamp(planmod.utc(end))
    folder.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": planmod.WINDOW_SCHEMA, "status": "READY",
               "window": {"end": end}, "data": {"sha256": "0" * 64}}
    if lead_class is not None:
        receipt["source"] = {"lead_class": lead_class}
    (folder / "ref.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_window_classes_and_the_oracle_refusal(tmp_path, table, anchor):
    c, b, e3d = table.arm("C"), table.arm("B"), table.arm("E3D")
    root = tmp_path
    windows = planmod.windows_root(root, anchor, c)
    for end in c.window_ends(anchor):
        _window(windows, planmod.iso(end), lead_class="forecast")
    record = planmod.check_arm_windows(root, anchor, c)
    assert record["classes"] == ["forecast"] and len(record["windows"]) == 12

    # One oracle window in the forecast arm's set.
    _window(windows, "2026-10-01T19:00:00Z", lead_class="oracle")
    with pytest.raises(PlanError, match="oracle window\\s+in a forecast column"):
        planmod.check_arm_windows(root, anchor, c)

    # Unlabelled windows (grid-ref writes none) are classed by time.
    b_windows = planmod.windows_root(root, anchor, b)
    for end in b.window_ends(anchor):
        _window(b_windows, planmod.iso(end))
    assert planmod.check_arm_windows(root, anchor, b)["classes"] == ["observed"]
    e_windows = planmod.windows_root(root, anchor, e3d)
    for end in e3d.window_ends(anchor):
        _window(e_windows, planmod.iso(end))
    assert planmod.check_arm_windows(root, anchor, e3d)["classes"] == ["oracle"]
    # ... and an "observed" label after t0 is still an oracle.
    receipt = {"window": {"end": "2026-10-01T18:15:00Z"}, "source": {"lead_class": "observed"}}
    assert planmod.classify_window(receipt, anchor.t0) == "oracle"
    # A "forecast" label is held to its own record: not causal, or issued
    # after t0, is an oracle whatever the label says.
    forecast = {"window": {"end": "2026-10-01T18:10:00Z"},
                "source": {"lead_class": "forecast", "causal": True,
                           "issue_time": "2026-10-01T18:00:00Z"}}
    assert planmod.classify_window(forecast, anchor.t0) == "forecast"
    late = copy.deepcopy(forecast)
    late["source"]["issue_time"] = "2026-10-01T18:10:00Z"
    assert planmod.classify_window(late, anchor.t0) == "oracle"
    peeked = copy.deepcopy(forecast)
    peeked["source"]["causal"] = False
    assert planmod.classify_window(peeked, anchor.t0) == "oracle"

    # A missing window.
    (b_windows / "20261001T1745Z" / "ref.json").unlink()
    with pytest.raises(PlanError, match="missing"):
        planmod.check_arm_windows(root, anchor, b)


def test_boundary_and_reference_inputs(table, anchor):
    rows = planmod.fetch_list(anchor, table.for_case(anchor, "full"), "full")
    names = [dest for _url, dest in rows]
    assert all(url.startswith("https://noaa-") for url, _dest in rows)
    assert "raw/hrrr.t18z.wrfnatf00.grib2" in names and "raw/hrrr.t17z.wrfnatf00.grib2" in names
    rap = sorted(int(n[-8:-6]) for n in names if "awp130bgrb" in n)
    assert rap == [2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23]
    refs = [n for n in names if n.startswith("reference/")]
    assert refs[0].endswith("wrfsfcf01.grib2") and refs[-1].endswith("wrfsfcf18.grib2")
    assert len(names) == len(set(names))


def test_scores_are_keyed_by_valid_time_after_t0(table, anchor):
    b = table.arm("B")
    first_heated_free_hour = b.start(anchor) + timedelta(hours=2)
    # B's own f02 is everyone's f01.
    assert anchor.lead_hours(first_heated_free_hour) == 1.0


def _history(root: Path, case, arm_id: str, hours, classes=()):
    from tools.heat_eval.score_case import seam
    out = root / case.id / "arms" / arm_id / "out" / "wrfout"
    out.mkdir(parents=True, exist_ok=True)
    for h in hours:
        # The door writes HH:MM:SS; underscores keep the fixture portable (the reader takes both).
        valid = seam(case.t0 + timedelta(hours=h)).replace("T", "_").replace(":", "_")
        (out / f"wrfout_d01_{valid}").write_bytes(b"")
    (root / case.id / "arms" / arm_id / "READY").write_text("x", encoding="utf-8")
    (root / case.id / "arms" / arm_id / "heating-classes.json").write_text(
        json.dumps({"classes": list(classes)}), encoding="utf-8")


def test_sheets_order_their_columns_and_refuse_an_oracle_in_a_forecast_column(tmp_path, monkeypatch, anchor):
    from tools.heat_eval import sheets
    monkeypatch.setenv("GPUWM_RW_COMPARE", "rw_compare")
    full = list(range(0, 19))
    _history(tmp_path, anchor, "A", full)
    _history(tmp_path, anchor, "B", range(-1, 7), ["observed"])   # core B stops at t0 + 6 h
    _history(tmp_path, anchor, "C", full, ["forecast"])
    _history(tmp_path, anchor, "E", full, ["oracle"])
    plans = {sheet["name"]: sheet for sheet in sheets.plan_sheets(tmp_path, anchor, "core")}
    early = plans["refc-arms-f01-f03"]
    assert early["columns"] == ["mrms", "1", "2", "3", "4", "hrrr"]
    assert early["arms"] == ["A", "B", "C", "E"]
    command = early["command"]
    panels = [command[i + 1] for i, token in enumerate(command) if token == "--panel"]
    assert len(panels) == 12 and command.count("--row") == 2
    assert panels[0].startswith("A: no heating=") and panels[0].endswith("wrfout_d01_2026-10-01_19_00_00")
    assert panels[1].startswith("B: ") and panels[1].endswith("wrfout_d01_2026-10-01_19_00_00")
    late = plans["refc-arms-f06-f18"]
    assert late["arms"] == ["A", "C", "E"] and late["skipped_arms"] == ["B"]
    assert plans["surface-t2m"]["columns"] == ["1", "2", "hrrr"]
    # An oracle-heated run in the forecast column C is refused.
    (tmp_path / anchor.id / "arms" / "C" / "heating-classes.json").write_text(
        json.dumps({"classes": ["forecast", "oracle"]}), encoding="utf-8")
    with pytest.raises(sheets.SheetRefused, match="only an oracle column"):
        sheets.plan_sheets(tmp_path, anchor, "core")


def test_each_arm_waits_for_its_own_preparation_and_its_windows(table):
    case = planmod.case_by_id("c1-20240521")
    assert table.arm("A").needs(case, "core") == ["prep-1800z-18h"]
    assert table.arm("B").needs(case, "core") == ["prep-1700z-7h", "windows-b"]
    assert table.arm("B").needs(case, "full") == ["prep-1700z-19h", "windows-b"]
    assert table.arm("C").needs(case, "core") == ["prep-1800z-18h", "windows-nowcast"]
    assert table.arm("C60").needs(case, "full") == ["prep-1800z-6h", "windows-nowcast"]
    assert table.arm("E").needs(case, "core") == ["prep-1800z-18h", "windows-oracle"]
    assert table.arm("E3D").needs(case, "full") == ["prep-1800z-6h", "windows-fwd"]
    with pytest.raises(PlanError, match="no case step writes"):
        planmod.arm_from_row(_row(heating={"windows": "nowhere", "window_minutes": 10,
                                           "active_minutes": 120})).needs(case, "core")


def test_g3_needs_echo_to_grow_and_never_gates_the_anchor(anchor):
    from tools.heat_eval import gates
    decision = planmod.case_by_id("c1-20240521")
    # No echo at t0 and none three hours later is not growth (it was a pass).
    assert gates.g3_growth(0, 0) == 0.0
    assert gates.g3_growth(0, 10) == math.inf
    assert gates.g3_growth(100, 150) == 1.5
    quiet = gates.g3_record(decision, {0: 0, 3: 0})
    assert quiet["measured_pass"] is False and quiet["open"] is False
    grows = gates.g3_record(decision, {0: 100, 3: 200})
    assert grows["open"] is True
    # The anchor is recorded, never gated: rule 6 needs no new storms.
    flat_anchor = gates.g3_record(anchor, {0: 100, 3: 100})
    assert flat_anchor["measured_pass"] is False and flat_anchor["open"] is True
    # A G3 that cannot be measured closes a decision case instead of hanging the chain.
    broken = gates.g3_record(decision, None, "OSError: no route")
    assert broken["open"] is False and broken["growth"] is None


def _junit(path: Path, cases) -> Path:
    rows = []
    for classname, name, outcome in cases:
        inner = {"pass": "", "fail": "<failure message='x'/>", "skip": "<skipped message='no cuda'/>",
                 "error": "<error message='x'/>"}[outcome]
        rows.append(f"<testcase classname='{classname}' name='{name}'>{inner}</testcase>")
    path.write_text("<testsuites><testsuite>" + "".join(rows) + "</testsuite></testsuites>",
                    encoding="utf-8")
    return path


def test_g0_does_not_pass_on_skipped_or_missing_gpu_tests(tmp_path):
    from tools.heat_eval import gates
    gpu = "tests.test_forecast_heating_gpu"
    ok = _junit(tmp_path / "ok.xml", [("tests.test_radar_tten_grid", "t1", "pass"),
                                      (gpu, "one_card_matches_two", "pass")])
    assert gates.g0_verdict(ok, 0)["pass"] is True
    # Every GPU test skipped: pytest exits 0, and the gate must still fail.
    skipped = _junit(tmp_path / "skip.xml", [("tests.test_radar_tten_grid", "t1", "pass"),
                                             (gpu, "one_card_matches_two", "skip")])
    record = gates.g0_verdict(skipped, 0)
    assert record["pass"] is False and record["required_file_skips"]
    # The GPU file absent (work package 2 not merged) fails too.
    absent = _junit(tmp_path / "absent.xml", [("tests.test_radar_tten_grid", "t1", "pass")])
    assert gates.g0_verdict(absent, 0)["pass"] is False
    assert gates.g0_verdict(tmp_path / "none.xml", 4)["pass"] is False
    failed = _junit(tmp_path / "fail.xml", [(gpu, "restart", "fail")])
    assert gates.g0_verdict(failed, 1)["pass"] is False


def _probe(tmp_path: Path, *, used_mib: float, reached: bool, frames: int = 2) -> Path:
    probe = tmp_path / "probe"
    (probe / "out" / "wrfout").mkdir(parents=True)
    (probe / "cards.txt").write_text("cards=2,3 n=2 start=x\n", encoding="utf-8")
    (probe / "gpu-samples.csv").write_text(
        f"2026/10/06 18:00:00.000, 2, {used_mib}, 99, 300\n"
        f"2026/10/06 18:00:02.000, 3, {used_mib - 1000}, 99, 300\n", encoding="utf-8")
    for hour in range(frames):
        (probe / "out" / "wrfout" / f"wrfout_d01_2026-10-01_{18 + hour:02d}_00_00").write_bytes(b"")
    (probe / "done.json").write_text(json.dumps({"rc": 0 if reached else 1, "reached_f01": reached}),
                                     encoding="utf-8")
    return probe


def test_g1_sets_four_cards_when_the_probe_did_not_reach_f01(tmp_path):
    from tools.heat_eval import gates
    fits = gates.g1(_probe(tmp_path / "a", used_mib=70 * 1024, reached=True))
    assert fits["cards_per_arm"] == 2 and fits["reached_f01"] is True
    # An out-of-memory death leaves samples under the limit; it must not read as "fits".
    died = gates.g1(_probe(tmp_path / "b", used_mib=70 * 1024, reached=False, frames=1))
    assert died["cards_per_arm"] == 4 and "did not reach f01" in died["note"]
    big = gates.g1(_probe(tmp_path / "c", used_mib=86 * 1024, reached=True))
    assert big["cards_per_arm"] == 4
    # The probe is arm A: the heating's slab and build transient (5.3 GiB) go on top.
    edge = gates.g1(_probe(tmp_path / "d", used_mib=81 * 1024, reached=True))
    assert edge["cards_per_arm"] == 4 and edge["heated_probe"] is False
    heated = gates.g1(_probe(tmp_path / "e", used_mib=81 * 1024, reached=True), heated=True)
    assert heated["cards_per_arm"] == 2


def test_station_rmse_is_read_as_a_number_from_the_scoreboard_result():
    from tools.heat_eval.score_case import scoreboard_rmse
    # The shape score_observations writes: metrics rows carry raw sums.
    sums = {"count": 4, "sumError": 0.0, "sumAbsoluteError": 4.0, "sumSquaredError": 9.0}
    result = {"metrics": [{"variable": "t2", "nativeVariable": "temperature_2m", "metric": "rmse",
                           "woof": sums}],
              "summary": [{"variable": "temperature_2m",
                           "woof": {"n": 4, "mae": 1.0, "rmse": 1.5, "bias": 0.0}}]}
    assert scoreboard_rmse(result, "t2") == 1.5
    assert scoreboard_rmse({"metrics": result["metrics"]}, "t2") == 1.5
    assert scoreboard_rmse(result, "td2") is None
    assert isinstance(scoreboard_rmse(result, "t2"), float)


def test_the_table_cut_is_the_doors_own(tmp_path, table, anchor):
    # On the box tree (work package 2 merged) the door's own split is here:
    # the preparation must hold exactly the bytes the door will check.
    heating = pytest.importorskip("gpuwm.da.forecast_heating")
    mac.write_case(tmp_path, anchor, table, "core", mac.TEMPLATE.read_text(encoding="utf-8"))
    for arm_id in ("A", "B", "C", "E"):
        text = (tmp_path / anchor.id / "arms" / arm_id / "experiment.toml").read_text(encoding="utf-8")
        assert heating.split_experiment_text(text)[0] == mac.table_free(text)
