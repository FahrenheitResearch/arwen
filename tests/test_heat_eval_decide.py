"""The nowcast heating decision rule on synthetic score tables.

Each rule is flipped on its own and the verdict must change with that rule
named; an oracle-heated run in a forecast column is refused; scores are
keyed by valid time so an arm that started an hour early lines up.
"""

from __future__ import annotations

import copy

import pytest

from tools.heat_eval import decide

LEADS = (1, 2, 3, 4, 5, 6, 12, 18)
DECISION = ("c1", "c2", "c3", "c4")


def _arm(fss: float, area: float, t2: float, td2: float, column: str, classes=()):
    return {
        "column": column, "lead_classes": list(classes),
        "by_lead": {str(lead): {"fss35_27km": fss, "area_ratio35": area,
                                "t2_rmse": t2, "td2_rmse": td2} for lead in LEADS},
    }


def _table(case_id: str, role: str = "decision") -> dict:
    return {
        "schema": decide.TABLE_SCHEMA, "case_id": case_id, "role": role,
        "registration_sha256": "f" * 64,
        "arms": {
            "A": _arm(0.20, 2.5, 1.50, 1.80, "A"),
            "C": _arm(0.24, 1.8, 1.52, 1.81, "C", ["forecast"]),
            "E": _arm(0.30, 1.2, 1.51, 1.80, "E", ["oracle"]),
        },
    }


def _tables():
    return [_table(case) for case in DECISION] + [_table("c5", "anchor")]


def _eye(better=(True, True, True, True)):
    return {"cases": {case: {"C_better_than_A": flag, "E_better_than_A": True}
                      for case, flag in zip(DECISION, better)}}


def _rule(result, name):
    return next(rule for rule in result["nowcast"]["rules"] if rule["rule"] == name)


def _set(tables, case, arm, lead, key, value):
    for table in tables:
        if table["case_id"] == case:
            table["arms"][arm]["by_lead"][str(lead)][key] = value


def test_a_clean_win_is_promoted_and_reports_the_nowcast_cost():
    result = decide.decide(_tables(), _eye())
    assert result["verdict"] == "promote", [r for r in result["nowcast"]["rules"] if r["pass"] is not True]
    assert result["rule_status"] == "proposed-unratified"
    assert all(abs(v - 0.06) < 1e-12 for v in result["nowcast_cost_fss35_27km_f01_f03"].values())
    assert result["oracle"]["verdict"] == "promote"


def test_without_the_eye_the_verdict_is_pending_never_a_pass():
    result = decide.decide(_tables(), None)
    assert result["verdict"] == "pending"
    assert _rule(result, "R1")["pass"] is None


@pytest.mark.parametrize("rule, change", [
    ("R1", lambda t: None),
    ("R2", lambda t: _set(t, "c2", "C", 2, "fss35_27km", 0.0)),
    ("R3", lambda t: _set(t, "c3", "C", 1, "area_ratio35", 3.0)),
    ("R4", lambda t: _set(t, "c4", "C", 4, "t2_rmse", 1.56)),
    ("R5", lambda t: _set(t, "c1", "C", 18, "fss35_27km", 0.16)),
    ("R6", lambda t: [_set(t, "c5", "C", lead, "fss35_27km", 0.15) for lead in (1, 2, 3)]),
])
def test_each_rule_flips_the_verdict_and_is_named(rule, change):
    tables = _tables()
    eye = _eye((True, False, False, True)) if rule == "R1" else _eye()
    change(tables)
    result = decide.decide(tables, eye)
    assert result["verdict"] == "do-not-promote"
    failed = [r["rule"] for r in result["nowcast"]["rules"] if r["pass"] is False]
    assert failed == [rule]


def test_the_area_rule_accepts_moving_toward_one_or_staying_in_band():
    tables = _tables()
    # 2.4 is outside [0.5, 2] but closer to 1 than A's 2.5: passes.
    _set(tables, "c1", "C", 1, "area_ratio35", 2.4)
    # 0.6 is farther from 1 than... A's 0.9 but inside the band: passes.
    _set(tables, "c2", "A", 2, "area_ratio35", 0.9)
    _set(tables, "c2", "C", 2, "area_ratio35", 0.6)
    assert _rule(decide.decide(tables, _eye()), "R3")["pass"] is True
    # 0.4 against A's 0.9: farther from 1 and outside the band.
    _set(tables, "c2", "C", 2, "area_ratio35", 0.4)
    assert _rule(decide.decide(tables, _eye()), "R3")["pass"] is False


def test_a_loss_beyond_the_margin_fails_r2_even_with_three_wins():
    tables = _tables()
    for lead in (1, 2, 3):
        _set(tables, "c4", "C", lead, "fss35_27km", 0.16)
    result = decide.decide(tables, _eye())
    assert _rule(result, "R2")["pass"] is False
    assert abs(_rule(result, "R2")["differences"]["c4"] + 0.04) < 1e-12


def test_a_missing_score_leaves_the_rule_pending():
    tables = _tables()
    del tables[0]["arms"]["C"]["by_lead"]["5"]
    result = decide.decide(tables, _eye())
    assert _rule(result, "R4")["pass"] is None
    assert result["verdict"] == "pending"


def test_an_oracle_heated_run_in_a_forecast_column_is_refused():
    tables = _tables()
    tables[1]["arms"]["C"]["lead_classes"] = ["forecast", "oracle"]
    with pytest.raises(decide.DecisionRefused, match="future observations"):
        decide.decide(tables, _eye())
    # The oracle column may hold it.
    decide.decide(_tables(), _eye())


def test_tables_from_different_registrations_are_refused():
    tables = _tables()
    tables[2]["registration_sha256"] = "0" * 64
    with pytest.raises(decide.DecisionRefused, match="different registrations"):
        decide.decide(tables, _eye())


def test_when_the_oracle_fails_too_the_method_is_blamed():
    tables = _tables()
    for table in tables:
        for arm in ("C", "E"):
            for lead in (1, 2, 3):
                table["arms"][arm]["by_lead"][str(lead)]["fss35_27km"] = 0.10
    result = decide.decide(tables, _eye())
    assert result["verdict"] == "do-not-promote"
    assert result["oracle"]["verdict"] == "do-not-promote"
    assert "heating method" in result["reading"]


def test_rows_are_keyed_by_valid_time_after_t0_for_an_arm_that_started_early():
    # Arm B starts at 17Z: its own leads 2 and 3 are valid 19Z and 20Z.
    rows = [{"valid": "2026-10-01T19:00:00Z", "own_lead": 2, "fss35_27km": 0.3},
            {"valid": "2026-10-01T20:00:00Z", "own_lead": 3, "fss35_27km": 0.2}]
    keyed = decide.key_by_valid("2026-10-01T18:00:00Z", rows)
    assert sorted(keyed) == ["1", "2"]
    assert keyed["1"]["own_lead"] == 2
    with pytest.raises(decide.DecisionRefused, match="whole hour"):
        decide.key_by_valid("2026-10-01T18:00:00Z", [{"valid": "2026-10-01T18:30:00Z"}])
    with pytest.raises(decide.DecisionRefused, match="two rows"):
        decide.key_by_valid("2026-10-01T18:00:00Z", rows + copy.deepcopy(rows[:1]))


def test_a_reserve_fills_the_slot_of_a_decision_case_that_failed_g3():
    tables = [_table(case) for case in DECISION[:3]] + [_table("c5", "anchor"),
                                                        _table("r1", "reserve")]
    admitted = decide.admit_tables(tables)
    roles = {t["case_id"]: (t["role"], t.get("stands_in", False)) for t in admitted}
    assert roles["r1"] == ("decision", True)
    eye = {"cases": {case: {"C_better_than_A": True, "E_better_than_A": True}
                     for case in (*DECISION[:3], "r1")}}
    result = decide.decide(admitted, eye)
    assert result["verdict"] == "promote"
    assert any(case["stands_in"] for case in result["cases"])
    # With all four decision cases present, a reserve table is left out.
    full = decide.admit_tables(_tables() + [_table("r1", "reserve")])
    assert "r1" not in {t["case_id"] for t in full}
