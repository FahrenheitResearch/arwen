"""Authoring, source authority and target-readiness agreement."""
from datetime import datetime
import json
import os
from pathlib import Path
import tomllib

import pytest

from gpuwm import cli, fetch, fetch_routes, runplan, source_adapters


def _author(tmp_path, source="era5", extra=()):
    out = tmp_path / "intent.toml"
    code = cli.main(["domain", "--source", source, "--point", "39,-98",
                     "--cycle", "2026-07-29T00", "--hours", "6",
                     "--vram-gib", "32", "--name", "intent", "--out", str(out), *extra])
    return code, out


@pytest.mark.parametrize("member", [0, 9])
def test_explicit_eda_authors_reloadable_product_and_clock(tmp_path, capsys, member):
    code, out = _author(tmp_path, extra=["--era5-product", "ensemble_members",
        "--era5-provider", "cds", "--member", str(member), "--cadence", "3"])
    assert code == 0
    raw = tomllib.loads(out.read_text())
    hints = raw["fetch"]
    assert {key: hints[key] for key in ("era5_product", "era5_provider", "member", "cadence", "retrieve")} == {
        "era5_product": "ensemble_members", "era5_provider": "cds", "member": member,
        "cadence": 3, "retrieve": True}
    fetch.validate_fetch_hints(hints, source=str(out))
    assert raw["case_data"]["forcing_interval_s"] == 10800
    assert "interval_seconds = 10800" in out.with_suffix(".namelist.wps").read_text()
    printed = capsys.readouterr().out
    assert "--era5-product ensemble_members" in printed
    assert "--era5-provider cds" in printed
    assert "--retrieve" in printed


def test_member_alone_keeps_reanalysis_and_names_explicit_product(tmp_path, capsys):
    code, out = _author(tmp_path, extra=["--member", "3"])
    assert code == 2
    assert not out.exists()
    text = capsys.readouterr().err
    assert "ensemble_members" in text and "reanalysis" in text
    assert "has no acquisition member axis" not in text


def test_printed_local_root_is_retained_in_intent(tmp_path):
    code, out = _author(tmp_path, source="20crv3")
    assert code == 0
    hints = tomllib.loads(out.read_text())["fetch"]
    assert Path(hints["source_root"]) == (out.parent / "data" / "intent").resolve()


@pytest.mark.parametrize("value", [True, "3", 3.0, float("nan"), float("inf")])
def test_local_cadence_has_common_spelling_refusal(value):
    local = [row.source_id for row in source_adapters.source_adapters()
             if runplan.drivability_for(row.source_id).get("requires_source_root")]
    assert local
    for name in local:
        with pytest.raises(ValueError, match="cadence"):
            fetch.validate_fetch_hints({"source": name, "cadence": value}, source="intent.toml")


def test_composition_cache_revalidates_same_stat_bytes(tmp_path, monkeypatch):
    from gpuwm import source_authorities as owner
    root = tmp_path / "authorities"
    root.mkdir()
    names = {role: role + ".json" for role in owner.PROFILE_ROLES}
    import hashlib
    for name in names.values():
        (root / name).write_text('{"value":1}')
    pins = {role: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for role, name in names.items()}
    monkeypatch.setattr(owner, "_AUTHORITY_ROOT", root)
    monkeypatch.setattr(owner, "_PACKAGED_PROFILES", {"fixture": {"files": names, "sha256": pins}})
    monkeypatch.setattr(owner, "_COMPOSITIONS", {})
    assert owner.packaged_composition("fixture")["value"] == 1
    path = root / names["composition"]
    stamp = path.stat()
    path.write_text('{"value":2}')
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(RuntimeError, match="hash differs"):
        owner.packaged_composition("fixture")


@pytest.mark.parametrize("root_state", ["absent", "missing", "empty"])
def test_check_and_go_refuse_the_same_missing_local_inputs(tmp_path, capsys, root_state):
    code, out = _author(tmp_path, source="20crv3")
    assert code == 0
    text = out.read_text()
    if root_state == "absent":
        text = "\n".join(line for line in text.splitlines() if not line.startswith("source_root =")) + "\n"
        out.write_text(text)
    elif root_state == "empty":
        from pathlib import Path
        Path(tomllib.loads(text)["fetch"]["source_root"]).mkdir(parents=True)
    capsys.readouterr()
    checked = cli.main(["check", str(out), "--vram-gib", "32", "--free-gib", "30"])
    check_error = capsys.readouterr().err
    launched = cli.main(["go", str(out), "--dry-run"])
    go_error = capsys.readouterr().err
    assert checked == launched == 2
    for message in (check_error, go_error):
        assert ("local input bytes" in message if root_state == "absent" else
                "does not exist" in message if root_state == "missing" else "no GRIB2 files" in message)


@pytest.mark.parametrize("member", ["10", "-1", "1.0", "true"])
def test_invalid_eda_member_stops_before_cycle_resolution(tmp_path, monkeypatch, member):
    def unexpected(*args, **kwargs):
        pytest.fail("publication queried for an invalid member")
    monkeypatch.setattr(fetch, "resolve_latest_cycle", unexpected)
    code, out = _author(tmp_path, extra=["--cycle", "latest", "--era5-product", "ensemble_members",
                                        "--member", member, "--cadence", "3"])
    assert code == 2
    assert not out.exists()


def test_latest_authoring_binds_member_and_one_cycle(tmp_path, monkeypatch):
    seen = []
    def resolve(source, last_hour, **selection):
        seen.append((source, last_hour, selection))
        return datetime(2026, 7, 29)
    monkeypatch.setattr(fetch, "resolve_latest_cycle", resolve)
    code, out = _author(tmp_path, source="gefs", extra=["--cycle", "latest", "--member", "p05"])
    assert code == 0
    raw = tomllib.loads(out.read_text())
    assert len(seen) == 1 and seen[0][2]["member"] == "p05"
    assert raw["fetch"]["cycle"] == "2026-07-29T00"
    argv = runplan._fetch_arguments_from_hints(raw["fetch"], out=tmp_path / "download")
    assert argv[argv.index("--cycle") + 1] == "2026-07-29T00"
    assert argv[argv.index("--member") + 1] == "p05"


def test_noah_component_override_retains_actual_coupling_constraints():
    from test_physics_registry import _single_plan
    from gpuwm.physics_registry import validate_physics_plan
    plan = _single_plan()
    plan["domains"][0]["components"] = {"land_surface": "noah-mp"}
    report = validate_physics_plan(plan)
    assert report["launchable"], report["errors"]
    plan["domains"][0]["components"]["surface_layer"] = "off"
    report = validate_physics_plan(plan)
    assert not report["launchable"]
    assert any("surface_layer" in str(error) for error in report["errors"])


def test_eda_default_provider_is_explicit_in_authored_intent(tmp_path):
    code, out = _author(tmp_path, extra=["--era5-product", "ensemble_members", "--member", "0"])
    assert code == 0
    hints = tomllib.loads(out.read_text())["fetch"]
    assert hints["era5_provider"] == "cds"
    assert hints["retrieve"] is True


def test_eda_manual_retrieval_refusal_agrees_at_both_doors(tmp_path):
    hints = {"source": "era5", "cycle": "2026-07-29T00", "hours": 6,
             "cadence": 3, "member": 0, "era5_product": "ensemble_members"}
    with pytest.raises(ValueError, match="native member verification"):
        fetch.validate_fetch_hints(hints, source="intent.toml")
    argv = runplan._fetch_arguments_from_hints(hints, out=tmp_path / "download")
    args = cli.build_parser().parse_args(["fetch", *argv])
    with pytest.raises(ValueError, match="native member verification"):
        fetch.fetch_main(args)
    assert not (tmp_path / "download").exists()


def test_native_cadence_is_accepted_while_missing_native_frames_are_refused(tmp_path, monkeypatch):
    class PublicationReached(Exception):
        pass
    monkeypatch.setattr(fetch, "require_published_cycle",
                        lambda *args, **kwargs: (_ for _ in ()).throw(PublicationReached()))
    rows = [row for row in source_adapters.source_adapters() if row.fetch_entire_window]
    assert rows
    for row in rows:
        cadence = int(row.forcing_interval_seconds / 3600)
        hints = {"source": row.source_id, "cycle": "2026-07-29T00", "hours": 6, "cadence": cadence}
        fetch.validate_fetch_hints(hints, source="intent.toml")
        argv = runplan._fetch_arguments_from_hints(hints, out=tmp_path / row.source_id)
        with pytest.raises(PublicationReached):
            fetch.fetch_main(cli.build_parser().parse_args(["fetch", *argv]))
        with pytest.raises(ValueError, match="every frame"):
            fetch.validate_fetch_hints({**hints, "cadence": cadence + 1}, source="intent.toml")


def test_staged_chain_verifies_and_consumes_the_selected_member_tree(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gpuwm import forcing_member, prep_handoff
    config = tmp_path / "case.toml"
    config.write_text('[fetch]\nsource="aigefs"\ncycle="2026-07-29T00"\nhours=6\nmember="mem007"\n')
    config.with_suffix(".namelist.wps").write_text("&share /\n")
    download = tmp_path / "download"
    download.mkdir()
    selected = download / "members" / "selected.txt"
    original = download / "upstream.txt"
    handoff = {"schema": fetch_routes.PREP_ARGUMENTS_SCHEMA,
               "source": "aigefs", "member": "mem007", "cycle": "2026-07-29T00",
               "argv": ["--source", "aigefs", "--input-list", str(original)],
               "unbound_supplement_roles": []}
    (download / fetch_routes.PREP_ARGUMENTS_NAME).write_text(json.dumps(handoff))
    monkeypatch.setattr(runplan, "_run_fetch", lambda *args, **kwargs: {})
    calls = []
    def select(document):
        calls.append("select")
        return ["--source", "aigefs", "--input-list", str(selected)]
    def verify(hints, document, **kwargs):
        calls.append("verify")
        assert document["argv"][-1] == str(selected)
        return {"input_list": str(selected)}
    monkeypatch.setattr(prep_handoff, "preparation_arguments", select)
    monkeypatch.setattr(forcing_member, "verify_handoff", verify)
    class Prepared(Exception):
        pass
    def prepared(receipt, root, prepare, **_notes):
        assert receipt["input_list"] == str(selected)
        return prepare()
    def stop(arguments):
        assert arguments[arguments.index("--input-list") + 1] == str(selected)
        raise Prepared()
    monkeypatch.setattr(forcing_member, "prepare_verified", prepared)
    monkeypatch.setattr(runplan, "_prepare_stage", lambda root, **kwargs: kwargs["run"]())
    monkeypatch.setattr(runplan, "_run_prep", stop)
    plan = SimpleNamespace(run_options={"data_dir": str(download), "geog_root": str(tmp_path)},
                           config_intent=None)
    observer = SimpleNamespace(enter_stage=lambda *a, **k: None, finish_stage=lambda *a, **k: None)
    # This member-selection fixture declares a stationary experiment; the
    # chain checks its relocation contract before invoking preparation.
    from gpuwm.experiment import RelocationConfig
    exp = SimpleNamespace(relocation=RelocationConfig(), domains=())
    with pytest.raises(Prepared):
        runplan._staged_chain(plan, config_path=config, exp=exp, observer=observer, run_dir=tmp_path / "run")
    assert calls == ["select", "verify"]


def test_unimplemented_selection_reports_implementation_before_route_permission():
    from copy import deepcopy
    from test_physics_registry import _single_plan
    from gpuwm.physics_registry import physics_registry, registry_sha256, validate_physics_plan
    registry = deepcopy(physics_registry())
    options = registry["components"]["pbl"]["options"]
    options["opaque-pending-option"] = {**options["ysu"], "implemented": False}
    plan = _single_plan()
    plan["registry_sha256"] = registry_sha256(registry)
    plan["domains"][0]["components"] = {"pbl": "opaque-pending-option"}
    report = validate_physics_plan(plan, registry=registry)
    assert not report["launchable"]
    assert not any(error["code"] == "component-override-route" for error in report["errors"])
    assert any("implement" in error["code"] for error in report["errors"])


def test_advisory_expert_profile_is_available_to_domain_authoring(tmp_path):
    from gpuwm.physics_compat import (
        NOAHMP_PROFILE_ID, MYNN_NOAHMP_PROFILE_ID,
        MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID,
    )
    from gpuwm.physics_menu import WIZARD_PHYSICS_PROFILES
    for profile in (NOAHMP_PROFILE_ID, MYNN_NOAHMP_PROFILE_ID,
                    MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID):
        assert profile in WIZARD_PHYSICS_PROFILES
    code, out = _author(tmp_path, extra=["--physics-profile", MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID])
    assert code == 0
    from gpuwm.domain_wizard import experiment_from_text
    assert experiment_from_text(out.read_text(), source=str(out)).root.run.sf_surface_physics == 4
