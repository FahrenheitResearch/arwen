"""Selected member checks use the real grammar and verifier, with synthetic inventory."""
import json
from pathlib import Path

import pytest

from gpuwm import forcing_member as fm
from gpuwm import member_prep
from gpuwm.member_grammar import MemberIdentityRefusal


def fixture(tmp_path, monkeypatch, source="gefs", member="p03", rows=None):
    adapter, grammar, selected = fm.member_contract(source, member)
    identity = grammar.member(selected)
    v = identity.verification
    row = {"index": "1", "pdt": str(v.product_definition_templates[0]),
           "member": str(identity.ordinal), "ensemble_type": str(v.type_of_ensemble_forecast),
           "ensemble_size": str(v.ensemble_size),
           "generating_process": str(v.type_of_generating_process or 0),
           "forecast_generating_process_id": str(v.forecast_generating_process_id or 0)}
    path = tmp_path / "not-a-member-name.grib2"
    path.write_bytes(b"synthetic payload; inventory is injected, not a GRIB fixture")
    inputs = tmp_path / "inputs.txt"
    inputs.write_text(str(path) + "\n")
    hints = {"source": source, "cycle": "2026-09-09T00", "member": member}
    handoff = {**hints, "argv": ["--input-list", str(inputs)]}
    monkeypatch.setattr(member_prep, "member_inventory_rows", lambda *a: rows if rows is not None else [row])
    return hints, handoff, path, row


@pytest.mark.parametrize("source,member", [("gefs", "c00"), ("gefs", "p30"),
                                           ("aigefs", "mem000"), ("aigefs", "mem030")])
def test_selected_member_verified_from_all_message_identity_octets(tmp_path, monkeypatch, source, member):
    hints, handoff, path, row = fixture(tmp_path, monkeypatch, source, member)
    receipt = fm.verify_handoff(hints, handoff, out=tmp_path)
    assert receipt["member"] == member
    assert receipt["files"][0]["messages"] == 1
    assert receipt["files"][0]["sha256"] == fm._digest(path)
    assert Path(receipt["input_list"]).read_text() == str(path) + "\n"
    fm.verify_unchanged(receipt)


@pytest.mark.parametrize("column,value", [("member", "4"), ("pdt", "2"),
    ("pdt", "0"), ("ensemble_size", "999"), ("ensemble_type", "99"),
    ("generating_process", "255"), ("forecast_generating_process_id", "255")])
def test_wrong_member_statistics_or_producing_system_never_prepares(tmp_path, monkeypatch, column, value):
    hints, handoff, _, row = fixture(tmp_path, monkeypatch)
    row[column] = value
    with pytest.raises(MemberIdentityRefusal):
        fm.verify_handoff(hints, handoff, out=tmp_path)
    assert not list(tmp_path.glob("member-verification-*.json"))


@pytest.mark.parametrize("key,value", [("member", "p04"), ("source", "aigefs"),
                                       ("cycle", "2026-09-09T06")])
def test_stale_handoff_rejected_before_native_inventory(tmp_path, monkeypatch, key, value):
    hints, handoff, _, _ = fixture(tmp_path, monkeypatch)
    handoff[key] = value
    monkeypatch.setattr(member_prep, "member_inventory_rows", lambda *a: pytest.fail("identity mismatch reached inventory"))
    with pytest.raises(MemberIdentityRefusal):
        fm.verify_handoff(hints, handoff, out=tmp_path)


def test_empty_inventory_is_not_a_verified_member(tmp_path, monkeypatch):
    hints, handoff, _, _ = fixture(tmp_path, monkeypatch, rows=[])
    with pytest.raises(MemberIdentityRefusal, match="no GRIB messages"):
        fm.verify_handoff(hints, handoff, out=tmp_path)


def test_changed_bytes_force_prepare_not_stale_member_reuse(tmp_path, monkeypatch):
    hints, handoff, path, _ = fixture(tmp_path, monkeypatch)
    root = tmp_path / "prep"
    builds = []
    def prepare():
        builds.append(root.exists())
        root.mkdir(exist_ok=True)
        return {"decision": "fixture"}
    first = fm.verify_handoff(hints, handoff, out=tmp_path)
    fm.prepare_verified(first, root, prepare)
    fm.prepare_verified(first, root, prepare)
    path.write_bytes(b"different synthetic member input")
    second = fm.verify_handoff(hints, handoff, out=tmp_path)
    assert first != second
    result = fm.prepare_verified(second, root, prepare)
    assert builds == [False, True, False]
    assert "member_input_superseded" in result
    assert json.loads((root / "forcing-member.json").read_text()) == second


def test_mutation_during_prepare_cannot_reach_forecast(tmp_path, monkeypatch):
    hints, handoff, path, _ = fixture(tmp_path, monkeypatch)
    receipt = fm.verify_handoff(hints, handoff, out=tmp_path)
    def mutate():
        path.write_bytes(b"replaced")
        return {}
    with pytest.raises(MemberIdentityRefusal, match="changed during preparation"):
        fm.prepare_verified(receipt, tmp_path / "prep", mutate)


def test_default_member_is_bound_and_both_ensembles_have_a_chain():
    from gpuwm.runplan import intent_drivability
    for source in ("gefs", "aigefs"):
        assert fm.member_contract(source)[2]
        assert intent_drivability()[source]["chain"] == "prepared:staged"


def test_staged_fetch_flag_preserves_member():
    from gpuwm.runplan import _fetch_arguments_from_hints
    args = _fetch_arguments_from_hints({"source": "aigefs", "member": "mem017"}, out=Path("data"))
    assert args[args.index("--member") + 1] == "mem017"


@pytest.mark.parametrize("source,member", [("gefs","p07"),("aigefs","mem007")])
@pytest.mark.parametrize("wrong_bytes", [False,True])
def test_actual_staged_chain_verifies_before_preparing(tmp_path,monkeypatch,source,member,wrong_bytes):
    from datetime import datetime
    from types import SimpleNamespace
    from gpuwm import runplan, fetch_routes, cyclone_setup, stage_cli
    from gpuwm.hrrr_prepared_bundle import render_wps_namelist

    data = tmp_path/"forcing"
    data.mkdir()
    hints, _, _, row = fixture(data,monkeypatch,source,member)
    if wrong_bytes:
        row["member"] = "8"
    out = tmp_path/"cyclone.toml"
    text, exp = cyclone_setup.configuration_text(cycle="2026090900",point=(18.,-65.),
                                                forcing_source=source,member=member,source=str(out))
    out.write_text(text)
    out.with_suffix(".namelist.wps").write_text(render_wps_namelist(exp))
    plan = SimpleNamespace(run_options={"data_dir":str(data),"geog_root":str(tmp_path)},
                           config_intent={})
    calls = []
    def fetch(arguments,*a,**kw):
        assert arguments[arguments.index("--member")+1] == member
        route = fetch_routes.resolve_request(source,cycle=datetime(2026,9,9),hours=6,
                                              member=member,out=data)
        for name in route.primary_files:
            p=data/name
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(b"fixture input for mocked native inventory")
        fetch_routes.write_handoff(route,data,donor_files={d.role:tmp_path/"donor" for d in route.donors})
        calls.append("fetch")
        return {}
    def prepare(arguments):
        inputs=Path(arguments[arguments.index("--input-list")+1])
        assert inputs.name.startswith("member-inputs-")
        Path(arguments[arguments.index("--output-root")+1]).mkdir(parents=True)
        calls.append("prepare")
    class Observer:
        def enter_stage(self,*a,**kw): pass
        def finish_stage(self,**kw): pass
        def arm_first_products(self,*a,**kw): pass
    monkeypatch.setattr(runplan,"_run_fetch",fetch)
    monkeypatch.setattr(runplan,"_run_prep",prepare)
    monkeypatch.setattr(runplan,"_chain_render_plan",lambda *a,**kw:{})
    monkeypatch.setattr(runplan,"_clear_forecast_output",lambda path,**kw:path)
    monkeypatch.setattr(runplan,"_staged_forecast",lambda *a,**kw:calls.append("forecast"))
    monkeypatch.setattr(runplan,"_chain_render",lambda *a,**kw:{"render_route":"unchanged"})
    monkeypatch.setattr(stage_cli,"resolve_bundle",lambda *a:{"layout":"single","source":source})
    monkeypatch.setattr(stage_cli,"sim_command",lambda *a,**kw:["python","-m","fixture"])
    if wrong_bytes:
        with pytest.raises(MemberIdentityRefusal):
            runplan._staged_chain(plan,config_path=out,exp=exp,observer=Observer(),run_dir=tmp_path/"run")
        assert calls == ["fetch"]
    else:
        runplan._staged_chain(plan,config_path=out,exp=exp,observer=Observer(),run_dir=tmp_path/"run")
        assert calls == ["fetch","prepare","forecast"]
        assert json.loads((tmp_path/"run/chain/prep/forcing-member.json").read_text())["member"] == member
