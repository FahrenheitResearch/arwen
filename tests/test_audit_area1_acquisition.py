"""Acquisition regression contracts."""
from datetime import datetime
import json
from pathlib import Path
import dataclasses

import pytest

from gpuwm import cli, fetch, fetch_routes, runplan
from gpuwm.filesystem_paths import canonical_path

CYCLE = datetime(2026, 7, 29, 0)


def _legacy_handoff(tmp_path, source="gdas"):
    tmp_path.mkdir(exist_ok=True)
    files = []
    for hour in (0, 3, 6):
        path = tmp_path / f"input.f{hour:03d}"
        path.write_bytes(b"GRIB" + bytes([hour]) + b"7777")
        files.append({"name": path.name, "sha256": fetch.sha256_file(path), "forecast_hour": hour})
    series = tmp_path / "series.json"
    series.write_text("{}")
    emitted = fetch._write_gfs_front_door_files(
        tmp_path, source=source, cycle=CYCLE, files=files, series=series)
    return files, emitted


def test_c010_legacy_handoff_is_bound_and_sealed(tmp_path):
    files, emitted = _legacy_handoff(tmp_path)
    path = tmp_path / fetch_routes.PREP_ARGUMENTS_NAME
    assert path.is_file()
    handoff = json.loads(path.read_text())
    assert handoff["source"] == handoff["prep_source"] == "gdas"
    assert handoff["cycle"] == "2026-07-29T00"
    argv = handoff["argv"]
    assert argv[:2] == ["--source", "gdas"]
    supplements = [argv[i+1] for i, flag in enumerate(argv) if flag == "--supplement"]
    assert supplements == [f"gdas_pgrb2_in_band_surface={tmp_path / row['name']}" for row in files]
    row = next(row for row in emitted if row["role"] == "prep-arguments")
    assert row["sha256"] == fetch.sha256_file(path)
    assert row["bytes"] == path.stat().st_size
    assert handoff["unbound_supplement_roles"] == []


def test_c010_and_c137_legacy_source_is_staged_by_default():
    assert runplan.prepared_chain_for_source("gdas") == "prepared:staged"


def test_c059_publication_limit_is_not_an_evidence_gate():
    message = fetch.gdas_capability_refusal(12)
    assert "publishes" in message and "f012" in message
    assert "certified" not in message and "corpus" not in message
    assert "stay inside --hours" in message and "--source gfs" in message


@pytest.mark.parametrize("source", ["gefs", "aigefs"])
def test_c008_member_routes_have_a_chain(source):
    assert runplan.prepared_chain_for_source(source) == "prepared:staged"


@pytest.mark.parametrize("source,member", [("gefs","p05"),("aigefs","mem007")])
def test_c008_member_hint_reaches_validation(source, member):
    fetch.validate_fetch_hints({"source":source, "member":member}, source="member.toml")


def test_c008_member_handoff_is_executable_not_just_text(tmp_path):
    plan = fetch_routes.resolve_request("aigefs", cycle=CYCLE, hours=6, member="mem007")
    fetch_routes.write_handoff(plan, tmp_path)
    handoff = json.loads((tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    step = handoff["member_prep"]
    assert step["member"] == "mem007"
    assert step["set"] == plan.member_set
    assert step["steps"] == list(plan.leads)
    assert canonical_path(step["inputs"]) == canonical_path(tmp_path / "upstream")
    assert canonical_path(step["output"]) == canonical_path(tmp_path / "members")
    assert canonical_path(step["input_list_after"]) != canonical_path(
        tmp_path / fetch_routes.INPUT_LIST_NAME)


def test_c008_intent_member_is_not_dropped():
    intent = {"point":"35,-97", "source":"gefs", "cycle":"2026-07-29T00", "member":"p03"}
    built = runplan._build_intent(intent, route="prepared")
    args = runplan.intent_arguments(built, out=Path("generated.toml"))
    assert args[args.index("--member")+1] == "p03"


def test_c008_invalid_member_is_refused_before_generation():
    with pytest.raises(ValueError, match="31 members"):
        runplan._build_intent({"point":"35,-97","source":"gefs",
                              "cycle":"2026-07-29T00","member":"p99"}, route="prepared")


def test_c008_deterministic_member_is_still_refused():
    with pytest.raises(ValueError, match="deterministic"):
        fetch.validate_fetch_hints({"source":"rap","member":"p03"}, source="request.toml")


@pytest.mark.parametrize("cadence", [2, 4, 12, 24])
def test_c038_reanalysis_accepts_published_hour_subsets(cadence):
    from gpuwm.era5_member import validate_selection
    assert validate_selection(cadence=cadence) is None
    times = fetch._era5_times(CYCLE, 24, cadence)
    assert len(times) == 24 // cadence + 1
    assert (times[1] - times[0]).total_seconds() == cadence * 3600


@pytest.mark.parametrize("cadence", [6, 12, 24])
def test_c038_member_accepts_subsets_of_its_native_clock(cadence):
    from gpuwm.era5_member import validate_selection
    assert validate_selection(product_type="ensemble_members", member=3,
                              cadence=cadence, cycle=CYCLE) == 3


def test_c038_coarse_warning_is_once_per_review_and_machine_readable(capsys):
    from gpuwm.era5_member import validate_selection
    from gpuwm.explain import explain_scope
    captured = []
    for _ in range(2):
        with explain_scope(), runplan.collect_warnings(captured):
            validate_selection(cadence=12)
            validate_selection(cadence=12)
    assert len(captured) == 2
    assert all("6-hour reference" in item["action"] for item in captured)
    assert capsys.readouterr().err.count("Boundary cadence is 12 hours") == 2


def test_c038_cli_does_not_override_source_cadence_grammar():
    from gpuwm.cli import build_parser
    args = build_parser().parse_args(["fetch", "--source", "era5", "--cadence", "12"])
    assert args.cadence == 12


def _member_inputs(tmp_path, monkeypatch, source="aigefs", member="mem007"):
    from gpuwm import member_prep
    from gpuwm.member_grammar import load_member_grammar
    from gpuwm.source_authorities import packaged_member_grammar
    plan = fetch_routes.resolve_request(source, cycle=CYCLE, hours=6, member=member)
    fetch_routes.write_handoff(plan, tmp_path)
    handoff = json.loads((tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    grammar = load_member_grammar(packaged_member_grammar(handoff["member_set"]))
    identity = grammar.member(member)
    verification = identity.verification
    row = {"index": "0", "pdt": str(verification.product_definition_templates[0]),
           "member": str(identity.ordinal),
           "ensemble_type": str(verification.type_of_ensemble_forecast),
           "ensemble_size": str(verification.ensemble_size),
           "generating_process": str(verification.type_of_generating_process),
           "forecast_generating_process_id": str(verification.forecast_generating_process_id),
           "derived_forecast": "-"}
    calls = []
    def inventory(path, executable=None):
        calls.append(Path(path))
        return [dict(row)]
    monkeypatch.setattr(member_prep, "member_inventory_rows", inventory)
    if handoff.get("member_prep"):
        for product in grammar.products():
            for step in plan.leads:
                path = tmp_path / "upstream" / grammar.relative_path(member, product, CYCLE, step)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"fixture {product} {step}".encode())
    else:
        argv = handoff["argv"]
        listing = Path(argv[argv.index("--input-list") + 1])
        for value in listing.read_text().splitlines():
            Path(value).parent.mkdir(parents=True, exist_ok=True)
            Path(value).write_bytes(b"fixture composed member bytes")
    return handoff, calls, row


def test_c008_consumer_stages_real_files_and_rechecks_receipt_on_reuse(tmp_path, monkeypatch):
    from gpuwm.prep_handoff import preparation_arguments
    from gpuwm import member_prep
    handoff, calls, _ = _member_inputs(tmp_path, monkeypatch)
    arguments = preparation_arguments(handoff)
    listing = Path(arguments[arguments.index("--input-list") + 1])
    files = [Path(value) for value in listing.read_text().splitlines()]
    assert len(files) == 4
    assert all(path.is_file() for path in files)
    assert len(calls) == 4
    receipt = next((tmp_path / "members").rglob(member_prep.RECEIPT_NAME))
    before = receipt.read_bytes()
    assert preparation_arguments(handoff) == arguments
    assert receipt.read_bytes() == before
    assert len(calls) == 8
    assert all(canonical_path(path).is_relative_to(canonical_path(tmp_path / "members"))
               for path in calls[4:])


@pytest.mark.parametrize("change", ["source", "staged", "extra", "grammar"])
def test_c008_consumer_refuses_changed_member_tree(tmp_path, monkeypatch, change):
    from gpuwm.prep_handoff import preparation_arguments
    from gpuwm import member_prep
    handoff, _, _ = _member_inputs(tmp_path, monkeypatch)
    args = preparation_arguments(handoff)
    listing = Path(args[args.index("--input-list") + 1])
    staged = Path(listing.read_text().splitlines()[0])
    if change == "source":
        upstream = next(path for path in (tmp_path / "upstream").rglob("*") if path.is_file())
        upstream.write_bytes(b"changed")
    elif change == "staged":
        staged.write_bytes(b"changed")
    elif change == "extra":
        staged.with_name("unlisted.bin").write_bytes(b"extra")
    else:
        receipt = next((tmp_path / "members").rglob(member_prep.RECEIPT_NAME))
        document = json.loads(receipt.read_text())
        document["member_set"]["sha256"] = "0" * 64
        receipt.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="differs|unexpected|identity"):
        preparation_arguments(handoff)


def test_c008_statistics_do_not_pass_as_a_member(tmp_path, monkeypatch):
    from gpuwm.prep_handoff import preparation_arguments
    handoff, _, row = _member_inputs(tmp_path, monkeypatch)
    row.update(pdt="2", derived_forecast="0")
    with pytest.raises(ValueError):
        preparation_arguments(handoff)
    assert not list((tmp_path / "members").rglob("*.partial"))


def test_c008_composed_member_inputs_are_verified_without_staging(tmp_path, monkeypatch):
    from gpuwm.prep_handoff import preparation_arguments
    handoff, calls, row = _member_inputs(tmp_path, monkeypatch, source="gefs", member="p05")
    args = preparation_arguments(handoff)
    assert args == handoff["argv"]
    assert calls
    row["member"] = "6"
    with pytest.raises(ValueError):
        preparation_arguments(handoff)


def _local_member_archive(root, hours=(0, 3, 6), member="001"):
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for hour in hours:
        for role in ("pl", "sfc"):
            path = root / f"mem{member}_20260729{hour:02d}_{role}.grb2"
            # Discovery's member authority is the archive filename.
            # Full GRIB field decoding remains the preparer's separate gate.
            path.write_bytes(b"GRIB" + bytes([hour]) + role.encode() + b"7777")
            files.append(path)
    return files


def test_c015_local_source_is_structurally_drivable_but_requires_a_root(tmp_path):
    verdict = runplan.intent_drivability()["20crv3"]
    assert verdict["routes"] == ["prepared"]
    assert verdict["requires_source_root"] is True
    with pytest.raises(ValueError, match="local input bytes"):
        runplan.prepared_chain_for_source("20crv3")
    assert runplan.prepared_chain_for_source("20crv3", source_root=tmp_path) == "prepared:staged"


def test_c015_local_hints_do_not_require_a_download_route(tmp_path):
    fetch.validate_fetch_hints({"source": "20crv3", "cycle": "2026-07-29T00",
        "hours": 6, "cadence": 3, "source_root": str(tmp_path)}, source="local.toml")


def test_c015_local_member_manifest_roundtrips_the_actual_reader(tmp_path):
    from gpuwm.local_preparation import inspect_local_inputs, publish_local_handoff
    from gpuwm.twentycrv3_direct import _manifest
    root = tmp_path / "archive"
    _local_member_archive(root)
    snapshot = inspect_local_inputs("20crv3", root, cycle=CYCLE, hours=6, cadence=3)
    path = publish_local_handoff(snapshot, tmp_path / "binding")
    handoff = json.loads(path.read_text())
    argv = handoff["argv"]
    manifest, files = _manifest(Path(argv[argv.index("--source-manifest") + 1]),
        argv[argv.index("--source-manifest-sha256") + 1])
    assert manifest["member"] == "001"
    assert manifest["cadence_seconds"] == 10800
    assert len(files) == 6
    assert "--author-input-manifest" not in argv
    assert "--source-root" not in argv
    assert publish_local_handoff(snapshot, tmp_path / "binding") == path


@pytest.mark.parametrize("fault", ["pair", "time", "member", "changed"])
def test_c015_local_input_refusals_are_before_preparation(tmp_path, fault):
    from gpuwm.local_preparation import inspect_local_inputs, publish_local_handoff
    root = tmp_path / "archive"
    files = _local_member_archive(root, hours=(0, 3) if fault == "time" else (0, 3, 6))
    if fault == "pair":
        files[-1].unlink()
    elif fault == "member":
        other = root / files[-1].name.replace("mem001", "mem002")
        files[-1].rename(other)
    if fault != "changed":
        with pytest.raises(ValueError):
            inspect_local_inputs("20crv3", root, cycle=CYCLE, hours=6, cadence=3)
    else:
        snapshot = inspect_local_inputs("20crv3", root, cycle=CYCLE, hours=6, cadence=3)
        files[0].write_bytes(b"changed")
        with pytest.raises(ValueError, match="changed after review"):
            publish_local_handoff(snapshot, tmp_path / "binding")


def test_c015_staged_chain_consumes_local_manifest_without_network(tmp_path, monkeypatch):
    from types import SimpleNamespace
    root = tmp_path / "archive"
    _local_member_archive(root)
    config = tmp_path / "case.toml"
    config.write_text('[fetch]\nsource="20crv3"\ncycle="2026-07-29T00"\n'
                      'hours=6\ncadence=3\n')
    config.with_suffix(".namelist.wps").write_text("&share /\n")
    plan = SimpleNamespace(run_options={"data_dir": str(root), "geog_root": str(tmp_path)},
                           config_intent=None)
    observer = SimpleNamespace(enter_stage=lambda *a, **k: None,
                               finish_stage=lambda *a, **k: None)
    def no_fetch(*a, **k):
        pytest.fail("local input attempted a network fetch")
    monkeypatch.setattr(runplan, "_run_fetch", no_fetch)
    seen = []
    class Prepared(Exception):
        pass
    def prep(arguments):
        seen.extend(arguments)
        raise Prepared()
    monkeypatch.setattr(runplan, "_run_prep", prep)
    # A stationary experiment: the chain reads its relocation contract
    # before it composes the preparation.
    from gpuwm.experiment import RelocationConfig
    exp = SimpleNamespace(relocation=RelocationConfig(), domains=())
    with pytest.raises(Prepared):
        runplan._staged_chain(plan, exp=exp, config_path=config,
                              run_dir=tmp_path / "run", observer=observer)
    assert seen[seen.index("--source") + 1] == "20crv3"
    assert "--source-manifest-sha256" in seen
    assert "--wps-namelist" in seen


def test_c050_shared_dispatcher_declares_every_chain():
    from gpuwm.source_cli import preparation_runners
    runners = preparation_runners()
    assert runners["twentycrv3_member_grib2_v1"].chain == "prepared:staged"
    assert all(callable(row.required) and callable(row.command) for row in runners.values())
    assert all(row.chain for row in runners.values())


def test_c035_local_mapped_source_names_the_missing_handoff(tmp_path):
    from gpuwm.local_preparation import inspect_local_inputs
    verdict = runplan.intent_drivability()["era5-l137"]
    assert verdict["requires_source_root"]
    with pytest.raises(ValueError, match="prep-arguments.json"):
        inspect_local_inputs("era5-l137", tmp_path, cycle=CYCLE, hours=6, cadence=3)


def test_c015_real_plan_review_binds_local_inputs(tmp_path):
    root = tmp_path / "archive"
    _local_member_archive(root)
    plan = runplan.build_plan({
        "schema": runplan.PLAN_SCHEMA, "name": "local-review", "route": "prepared",
        "config": {"intent": {"point": "39,-98", "source": "20crv3",
            "cycle": "2026-07-29T00", "hours": 6, "vram_gib": 32, "data_dir": str(root)}}},
        source="local-plan.json", base_dir=tmp_path, sha256="0" * 64)
    document, _, _ = runplan.resolve_plan(plan)
    matches = [row for row in document["automatic_resolutions"] if row["key"] == "local_inputs"]
    assert len(matches) == 1
    assert matches[0]["value"]["file_count"] == 6
    assert matches[0]["value"]["source_root"] == str(root)
    assert not (root / "source-manifest.json").exists()


def test_c015_real_plan_review_refuses_absent_local_directory(tmp_path):
    plan = runplan.build_plan({
        "schema": runplan.PLAN_SCHEMA, "name": "local-review", "route": "prepared",
        "config": {"intent": {"point": "39,-98", "source": "20crv3",
            "cycle": "2026-07-29T00", "hours": 6, "vram_gib": 32}},
        "run_options": {"data_dir": str(tmp_path / "absent")}},
        source="local-plan.json", base_dir=tmp_path, sha256="0" * 64)
    with pytest.raises(ValueError, match="Local source directory"):
        runplan.resolve_plan(plan)


def test_c015_real_plan_review_refuses_a_missing_root_before_execution(tmp_path):
    plan = runplan.build_plan({
        "schema": runplan.PLAN_SCHEMA, "name": "local-review", "route": "prepared",
        "config": {"intent": {"point": "39,-98", "source": "20crv3",
            "cycle": "2026-07-29T00", "hours": 6, "vram_gib": 32}}},
        source="local-plan.json", base_dir=tmp_path, sha256="0" * 64)
    with pytest.raises(ValueError, match="does not exist"):
        runplan.resolve_plan(plan)


@pytest.mark.parametrize("provider", ["cds", "arco"])
def test_c038_retrieval_reaches_transport_with_coarser_valid_clock(tmp_path, monkeypatch, provider):
    class TransportReached(Exception):
        pass
    if provider == "cds":
        from gpuwm import era5_acquisition
        def stop(progress):
            raise TransportReached()
        monkeypatch.setattr(era5_acquisition, "_client", stop)
        acquire = era5_acquisition.retrieve_era5
    else:
        from gpuwm import era5_arco, zarr_bridge
        def stop(request, **kwargs):
            assert request["times"] == ["2026-07-29T00:00:00Z",
                                       "2026-07-29T12:00:00Z", "2026-07-30T00:00:00Z"]
            raise TransportReached()
        monkeypatch.setattr(zarr_bridge, "extract_regular_zarr", stop)
        acquire = era5_arco.retrieve_era5_arco
    with pytest.raises(TransportReached):
        acquire(cycle=CYCLE, hours=24, cadence=12, area="30,-100,40,-90",
                out=tmp_path / provider, progress=lambda _: None)


def test_c008_a_member_axis_is_not_added_to_a_deterministic_native_source():
    with pytest.raises(ValueError, match="no acquisition member axis"):
        fetch.validate_fetch_hints({"source": "gfs", "member": "p01"}, source="request.toml")


def test_c008_real_intent_emission_preserves_member_in_config_and_next_command(tmp_path, capsys):
    plan = runplan.build_plan({
        "schema": runplan.PLAN_SCHEMA, "name": "member-review", "route": "prepared",
        "config": {"intent": {"point": "39,-98", "source": "gefs",
            "cycle": "2026-07-29T00", "hours": 6, "vram_gib": 32, "member": "p05"}}},
        source="member-plan.json", base_dir=tmp_path, sha256="0" * 64)
    document, _, _ = runplan.resolve_plan(plan)
    selected = [row for row in document["automatic_resolutions"] if row["key"] == "member"]
    assert any(row["value"]["id"] == "p05" for row in selected)
    import tomllib
    assert tomllib.loads(document["generated_config"])["fetch"]["member"] == "p05"
    assert "gpuwm go " in capsys.readouterr().out


def test_c035_existing_local_handoff_checks_required_roles_and_file_bytes(tmp_path):
    from gpuwm.local_preparation import inspect_local_inputs, publish_local_handoff
    root = tmp_path / "input"
    root.mkdir()
    primary = root / "primary.nc"
    donor = root / "invariant.nc"
    primary.write_bytes(b"synthetic primary bytes")
    donor.write_bytes(b"synthetic invariant bytes")
    listing = root / "input-list.txt"
    listing.write_text(str(primary) + "\n")
    role = "twentycrv3_netcdf_recovered_invariant"
    tokens = ["--source", "20crv3-cf", "--input-list", str(listing),
              "--supplement", f"{role}={donor}",
              "--author-input-manifest", str(root / "inputs.json")]
    handoff = fetch_routes.write_prep_arguments(
        root, source="20crv3-cf", prep_source="20crv3-cf", cycle=CYCLE, tokens=tokens)
    snapshot = inspect_local_inputs("20crv3-cf", root, cycle=CYCLE, hours=6, cadence=3)
    assert len(snapshot["files"]) == 2
    published = publish_local_handoff(snapshot, tmp_path / "run")
    assert json.loads(published.read_text())["source"] == "20crv3-cf"
    document = json.loads(handoff.read_text())
    index = document["argv"].index("--supplement")
    del document["argv"][index:index+2]
    handoff.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="supplement roles differ"):
        inspect_local_inputs("20crv3-cf", root, cycle=CYCLE, hours=6, cadence=3)


# ---------------------------------------------------------------------------
# The cadence a shared flag no longer constrains, checked per source
# ---------------------------------------------------------------------------

def test_a_gdas_cadence_that_does_not_divide_the_window_is_refused():
    """A cadence off the window truncated the series in silence.

    The shared ``--cadence`` flag stopped naming three values, because
    ERA5 accepts any whole hour.  GDAS took the same freedom and planned
    ``range(start, start + hours + 1, cadence)``, which simply stops
    short when the cadence does not divide the window: the fetch
    succeeded, its manifest was complete for what it held, and the run
    was bounded by a shorter series than the one asked for.
    """

    assert fetch.gdas_forecast_hours(6, 3) == (0, 3, 6)
    assert fetch.gdas_forecast_hours(9, 3) == (0, 3, 6, 9)
    assert fetch.gdas_forecast_hours(8, 2) == (0, 2, 4, 6, 8)
    assert fetch.gdas_forecast_hours(0, 1) == (0,)
    with pytest.raises(ValueError) as refusal:
        fetch.gdas_forecast_hours(9, 2)
    message = str(refusal.value)
    assert "--cadence 2" in message and "--hours 9" in message
    # The remedy names the cadences the ladder actually serves for this
    # window, derived from it: 9 divides by 1, 3 and 9 and every lead of
    # each lands on the published hours.
    assert "1 or 3 or 9" in message
    assert "f008" in message and "f009" in message


def test_a_gdas_cadence_is_derived_from_the_ladder_not_a_literal(monkeypatch):
    """Widen the published ladder and the accepted cadences widen with it."""

    monkeypatch.setattr(fetch, "GDAS_MAX_FORECAST_HOUR", 12)
    monkeypatch.setattr(fetch, "GDAS_PUBLISHED_HOURS", tuple(range(13)))
    assert fetch.gdas_forecast_hours(12, 4) == (0, 4, 8, 12)
    with pytest.raises(ValueError) as refusal:
        fetch.gdas_forecast_hours(10, 4)
    assert "1 or 2 or 5 or 10" in str(refusal.value)


def test_a_both_doors_plan_one_gdas_window(tmp_path, capsys):
    """The config table and the flag ask the same function.

    The config door planned every container source on the GFS ladder,
    and only when the cadence was one of two values it named itself, so
    a GDAS table carrying any other cadence was accepted at load and
    would have been refused at the fetch.
    """

    fetch.validate_fetch_hints(
        {"source": "gdas", "hours": 8, "cadence": 2}, source="gdas.toml")
    with pytest.raises(ValueError) as config_refusal:
        fetch.validate_fetch_hints(
            {"source": "gdas", "hours": 9, "cadence": 2}, source="gdas.toml")
    assert "--cadence 2" in str(config_refusal.value)

    rc = cli.main(["fetch", "--source", "gdas", "--cycle", "2026-07-29T00",
                   "--hours", "9", "--cadence", "2",
                   "--area", "30,-100,40,-90",
                   "--out", str(tmp_path / "out")])
    assert rc == 2
    assert "--cadence 2" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("cadence", [4, 5, 7])
def test_a_era5_keeps_any_whole_hour(cadence):
    """The freedom the shared flag was widened for is not taken back."""

    fetch.validate_fetch_hints(
        {"source": "era5", "hours": cadence * 3, "cadence": cadence},
        source="era5.toml")
    assert len(fetch._era5_times(CYCLE, cadence * 3, cadence)) == 4


# ---------------------------------------------------------------------------
# What a local-input configuration may not carry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("crop", [
    {"area": "30,-100,40,-90"},
    {"point": "35,-97", "radius_km": 500},
])
def test_b_a_crop_on_a_local_input_config_is_refused_by_name(tmp_path, crop):
    """Accepted and ignored is the defect; the key is named instead."""

    table = {"source": "20crv3", "cycle": "2026-07-29T00", "hours": 6,
             "cadence": 3, "source_root": str(tmp_path), **crop}
    with pytest.raises(ValueError) as refusal:
        fetch.validate_fetch_hints(table, source="local.toml")
    message = str(refusal.value)
    for key in sorted(crop):
        assert key in message
    assert "20crv3" in message
    assert "remove" in message and "gpuwm prep" in message


def test_b_source_root_on_a_downloaded_source_is_refused_by_name(tmp_path):
    """The mirror image of the same defect, in the same words."""

    with pytest.raises(ValueError) as refusal:
        fetch.validate_fetch_hints(
            {"source": "gfs", "cycle": "2026-07-29T00", "hours": 6,
             "area": "30,-100,40,-90", "source_root": str(tmp_path)},
            source="download.toml")
    message = str(refusal.value)
    assert "source_root" in message and "--data-dir" in message


def test_b_a_crop_a_download_route_refuses_is_still_refused():
    """Retiring the local-input skip does not retire the whole-object gate."""

    whole_object = sorted(
        source for source in fetch._fetch_hint_sources()
        if not fetch.fetch_accepts_area(source))
    assert whole_object
    with pytest.raises(ValueError, match="names a crop"):
        fetch.validate_fetch_hints(
            {"source": whole_object[0], "cycle": "2026-07-29T00", "hours": 6,
             "area": "30,-100,40,-90"}, source="whole-object.toml")


# ---------------------------------------------------------------------------
# The publication limit, layered
# ---------------------------------------------------------------------------

def test_c_the_publication_limit_is_layered_into_action_and_why():
    """The remedy at the default width, the mechanism behind --explain."""

    from gpuwm.explain import EXPLAIN_MARK, split

    message = fetch.gdas_capability_refusal(12)
    assert EXPLAIN_MARK in message
    action, why = split(message)
    assert "f012" in action and "f009" in action
    assert "What to do" in action and "--source gfs" in action
    assert "Why" in why and "never written" in why
    # The corrected reason: a publication boundary, not an evidence gate.
    assert "certified" not in message and "corpus" not in message


# ---------------------------------------------------------------------------
# One canonicalizing lookup for the drivability verdict
# ---------------------------------------------------------------------------

def test_d_a_source_alias_reaches_the_same_local_admission():
    """An alias read as "no verdict" and skipped the local admission."""

    from gpuwm.source_adapters import get_source_adapter

    canonical = runplan.drivability_for("20crv3")
    assert canonical["requires_source_root"] is True
    for alias in get_source_adapter("20crv3").aliases:
        assert runplan.drivability_for(alias) == canonical, alias
    assert runplan.drivability_for("not-a-source") == {}


def test_d_an_alias_config_is_admitted_as_a_local_input_plan(tmp_path):
    """A config spelling an alias gets the local admission, not a download.

    The intent route canonicalizes on its way in, so the gap showed only
    where a config carries the alias itself: review read no verdict for
    it, skipped the local input admission entirely, and left the staged
    chain to go looking for an acquisition route that does not exist.
    """

    root = tmp_path / "archive"
    _local_member_archive(root)
    config = _generated_local_config(tmp_path, root)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            'source = "20crv3"', 'source = "twentycrv3"'),
        encoding="utf-8")
    assert 'source = "twentycrv3"' in config.read_text(encoding="utf-8")

    plan = runplan.build_plan({
        "schema": runplan.PLAN_SCHEMA, "name": "alias-review",
        "route": "prepared", "config": {"path": str(config)},
        "run_options": {"data_dir": str(root)}},
        source="alias-plan.json", base_dir=tmp_path, sha256="0" * 64)
    document, _, _ = runplan.resolve_plan(plan)
    bound = [row for row in document["automatic_resolutions"]
             if row["key"] == "local_inputs"]
    assert len(bound) == 1
    assert bound[0]["value"]["source_root"] == str(root)


# ---------------------------------------------------------------------------
# A [fetch] table missing what a local input is selected by
# ---------------------------------------------------------------------------

def _generated_local_config(tmp_path, root) -> Path:
    """The config the intent route writes for a local-input source."""

    generated = tmp_path / "generated"
    generated.mkdir(exist_ok=True)
    plan = runplan.build_plan({
        "schema": runplan.PLAN_SCHEMA, "name": "local-review",
        "route": "prepared",
        "config": {"intent": {"point": "39,-98", "source": "20crv3",
                              "cycle": "2026-07-29T00", "hours": 6,
                              "vram_gib": 32, "data_dir": str(root)}}},
        source="local-plan.json", base_dir=tmp_path, sha256="0" * 64)
    runplan.resolve_plan(plan, generate_into=generated)
    return next(generated.glob("*.toml"))


def test_e_a_local_input_config_missing_a_key_is_a_refused_plan(tmp_path):
    """Not a KeyError from inside the review: the missing line is named."""

    import re

    root = tmp_path / "archive"
    _local_member_archive(root)
    config = _generated_local_config(tmp_path, root)
    text = config.read_text(encoding="utf-8")
    stripped = re.sub(r"(?m)^cycle = .*\n", "", text, count=1)
    assert stripped != text
    config.write_text(stripped, encoding="utf-8")

    from_file = runplan.build_plan({
        "schema": runplan.PLAN_SCHEMA, "name": "local-review",
        "route": "prepared", "config": {"path": str(config)},
        "run_options": {"data_dir": str(root)}},
        source="local-plan.json", base_dir=tmp_path, sha256="0" * 64)
    with pytest.raises(runplan.PlanError) as refusal:
        runplan.resolve_plan(from_file)
    message = str(refusal.value)
    assert "cycle" in message and "[fetch]" in message
    assert "what to do" in message


# ---------------------------------------------------------------------------
# Module boundaries
# ---------------------------------------------------------------------------

def test_f_the_new_modules_reach_only_public_names():
    """A private name imported across a module is a boundary that moved."""

    import ast
    import gpuwm

    root = Path(gpuwm.__file__).parent
    borrowed = []
    for name in ("prep_handoff.py", "local_preparation.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("gpuwm"):
                borrowed += [f"{name}: {node.module}.{alias.name}"
                             for alias in node.names
                             if alias.name.startswith("_")]
            if isinstance(node, ast.Attribute) and node.attr.startswith("_") \
                    and isinstance(node.value, ast.Name) \
                    and node.value.id in {"fetch", "fetch_routes", "runplan"}:
                borrowed.append(f"{name}: {node.value.id}.{node.attr}")
    assert borrowed == []


def test_f_the_tables_own_acquisition_refusal_is_readable_publicly():
    """The run plan quotes the table's sentence instead of copying it."""

    reason = fetch_routes.acquisition_refusal_reason("20crv3")
    assert reason and reason == str(fetch_routes.acquisition_refusal("20crv3")["why"])
    assert reason == runplan.drivability_for("20crv3")["source_root_reason"]
    # An alias asks the same question, and an unrefused source still answers.
    assert fetch_routes.acquisition_refusal_reason("twentycrv3") == reason
    # A source the fetch door serves has no absence to explain.
    assert fetch_routes.acquisition_refusal("gfs") is None
    assert fetch_routes.acquisition_refusal_reason("gfs") == ""


# ---------------------------------------------------------------------------
# What plan review reads off disk
# ---------------------------------------------------------------------------

def test_g_a_packaged_composition_is_parsed_once_but_reverified(monkeypatch):
    from gpuwm import source_authorities as owner
    profile = owner.packaged_profile_ids()[0]
    parsed = []
    loads = owner.json.loads
    def counted(data):
        parsed.append(data)
        return loads(data)
    monkeypatch.setattr(owner.json, "loads", counted)
    monkeypatch.setattr(owner, "_COMPOSITIONS", {})
    first = owner.packaged_composition(profile)
    assert owner.packaged_composition(profile) is first
    assert len(parsed) == 1


def test_g_a_rewritten_authority_requires_a_matching_pin(monkeypatch, tmp_path):
    import hashlib
    from gpuwm import source_authorities as owner
    names = {role: role + ".json" for role in owner.PROFILE_ROLES}
    pins = {}
    for role, name in names.items():
        path = tmp_path / name
        path.write_text('{"value":1}')
        pins[role] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(owner, "_AUTHORITY_ROOT", tmp_path)
    monkeypatch.setattr(owner, "_PACKAGED_PROFILES", {"fixture": {"files": names, "sha256": pins}})
    monkeypatch.setattr(owner, "_COMPOSITIONS", {})
    first = owner.packaged_composition("fixture")
    path = tmp_path / names["composition"]
    path.write_text('{"value":2}')
    with pytest.raises(RuntimeError, match="hash differs"):
        owner.packaged_composition("fixture")
    pins["composition"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert owner.packaged_composition("fixture")["value"] == 2
    assert first["value"] == 1


# ---------------------------------------------------------------------------
# Where the handoff binding is checked
# ---------------------------------------------------------------------------

def test_h_a_container_without_an_in_band_binding_refuses_before_the_download(
        tmp_path, monkeypatch, capsys):
    """The refusal used to arrive after the bytes, inside the publisher."""

    def never(*args, **kwargs):
        raise AssertionError("the transport was reached")

    monkeypatch.setattr(fetch_routes, "in_band_supplement_role",
                        lambda source: None)
    monkeypatch.setattr(fetch, "require_published_cycle", never)
    monkeypatch.setattr(fetch, "fetch_gfs", never)
    out = tmp_path / "out"
    rc = cli.main(["fetch", "--source", "gdas", "--cycle", "2026-07-29T00",
                   "--hours", "6", "--area", "30,-100,40,-90",
                   "--out", str(out)])
    assert rc == 2
    message = capsys.readouterr().err
    assert "in-band surface binding" in message
    assert "gpuwm prep --source gdas" in message
    assert not out.exists()


def test_h_the_publisher_asks_the_same_function(tmp_path, monkeypatch):
    """One binding check, so the two doors cannot disagree about a source."""

    monkeypatch.setattr(fetch_routes, "in_band_supplement_role",
                        lambda source: None)
    with pytest.raises(ValueError, match="in-band surface binding"):
        fetch.container_handoff_binding("gdas")
    with pytest.raises(ValueError, match="in-band surface binding"):
        _legacy_handoff(tmp_path / "publish")


# ---------------------------------------------------------------------------
# The config's own input root has to reach the door that runs it
# ---------------------------------------------------------------------------

def _root_only_config(tmp_path, root) -> Path:
    """The emitted local-input config, carrying [fetch].source_root alone."""

    config = _generated_local_config(tmp_path, root)
    assert "source_root" in config.read_text(encoding="utf-8")
    return config


def test_r_a_config_root_alone_drives_the_go_door(tmp_path):
    """`[fetch].source_root` is the whole root a local-input config needs.

    `gpuwm go` computed a managed download directory for every config
    carrying a source and a cycle, and the local-input resolution takes
    the run option whenever it is set, so the key the wizard writes, and
    the only root a config can carry by itself, was buried by a
    directory the reader never named.  A source with no acquisition
    route downloads nothing, so there is no managed cache to compute.
    """

    from gpuwm.cli import main as cli_main

    root = tmp_path / "archive"
    _local_member_archive(root)
    config = _root_only_config(tmp_path, root)
    assert cli_main(["go", str(config), "--dry-run"]) == 0


def test_r_a_local_config_with_no_root_names_the_keys_it_wants(tmp_path, capsys):
    """The refusal names the config key, never a path the reader never typed."""

    import re

    from gpuwm.cli import main as cli_main

    root = tmp_path / "archive"
    _local_member_archive(root)
    config = _root_only_config(tmp_path, root)
    text = config.read_text(encoding="utf-8")
    stripped = re.sub(r"(?m)^source_root = .*\n", "", text, count=1)
    assert stripped != text
    config.write_text(stripped, encoding="utf-8")

    assert cli_main(["go", str(config), "--dry-run"]) == 2
    error = capsys.readouterr().err
    assert "[fetch].source_root" in error and "--data-dir" in error
    assert "downloads" not in error
    assert "Traceback" not in error


def test_r_an_explicit_data_dir_still_wins_over_the_config(tmp_path):
    """The flag the reader typed outranks the key the wizard wrote."""

    from gpuwm.cli import main as cli_main

    stale = tmp_path / "stale"
    _local_member_archive(stale)
    config = _root_only_config(tmp_path, stale)
    fresh = tmp_path / "fresh"
    _local_member_archive(fresh)
    config.write_text(
        config.read_text(encoding="utf-8").replace(str(stale), str(tmp_path / "gone")),
        encoding="utf-8")
    assert cli_main(["go", str(config), "--dry-run",
                     "--data-dir", str(fresh)]) == 0


# ---------------------------------------------------------------------------
# Shipped prose may not deny a route the registry declares
# ---------------------------------------------------------------------------

#: Sentences that state a named source has no preparation or ingest route.
_DENIALS = (r"no ingest route", r"no initialization front door",
            r"fetch and decode only", r'"runnable":\s*false',
            r"front door at all", r"there is no [a-z0-9_-]+ front door",
            r"does not imply a [a-z0-9_-]+ ingest route")


def _paragraphs(text: str) -> list:
    return [block for block in text.split("\n\n") if block.strip()]


@pytest.mark.parametrize("page", ["DATA.md", "SOURCES.md", "CLI-USER-MANUAL.md"])
def test_r_a_shipped_page_denies_no_route_the_registry_declares(page):
    """A retired refusal that survives in prose still refuses the reader.

    The denial is not looked for by source name: every registry id and
    alias named in a paragraph that denies a route is checked against
    the SAME drivability verdict the doors read, so the next source to
    gain a runner cannot leave a contradiction behind in a shipped page.
    """

    import re

    from gpuwm.source_adapters import source_adapters

    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "public" / page).read_text(encoding="utf-8")
    names = {}
    for adapter in source_adapters():
        for spelling in (adapter.source_id, *getattr(adapter, "aliases", ())):
            names[str(spelling).lower()] = adapter.source_id
    offenders = []
    for block in _paragraphs(text):
        lowered = block.lower()
        if not any(re.search(phrase, lowered) for phrase in _DENIALS):
            continue
        for spelling, source_id in names.items():
            if not re.search(r"(?<![a-z0-9_-])" + re.escape(spelling)
                             + r"(?![a-z0-9_-])", lowered):
                continue
            verdict = runplan.drivability_for(source_id)
            if str(verdict.get("chain") or "").startswith("prepared:"):
                offenders.append((source_id, block.strip().splitlines()[0]))
    assert not offenders, (
        f"docs/public/{page} denies a route the registry declares: {offenders}")


# ---------------------------------------------------------------------------
# The flag's help may not assert the behaviour the flag no longer has
# ---------------------------------------------------------------------------

def _source_help(parser) -> str:
    """The ``--source`` help string of the domain wizard's parser."""

    stack = [parser]
    while stack:
        current = stack.pop()
        for action in getattr(current, "_actions", ()):
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                stack.extend(choices.values())
            if ("--source" in getattr(action, "option_strings", ())
                    and (action.help or "").startswith("forcing source:")):
                return action.help or ""
    raise AssertionError("the domain wizard has no --source help")


def test_r_the_source_help_describes_the_table_a_local_source_now_gets():
    """The wizard emits a [fetch] table for a local-input source.

    The help said such a source gets the acquisition step named INSTEAD
    of a table, which is what it used to do, and the shipped
    CLI-OPTIONS page copies this string verbatim.
    """

    from gpuwm.cli import build_parser

    help_text = " ".join(_source_help(build_parser()).split())
    assert "instead of a [fetch] table" not in help_text
    assert "local input contract" in help_text
    assert "source_root" in help_text


# ---------------------------------------------------------------------------
# One window planner, whatever the cadence is spelled as
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cadence", [4.0, "4"])
def test_r_a_cadence_spelling_cannot_skip_the_window_check(cadence):
    """A decimal or quoted cadence was read, accepted, and refused later.

    The scalar check admits a float and a string, and the window check
    ran only for a plain integer, so the two doors the lane joined
    disagreed again for exactly those two spellings.
    """

    table = {"source": "gdas", "cycle": "2026-07-29T00", "hours": 6,
             "cadence": cadence}
    with pytest.raises(ValueError) as refusal:
        fetch.validate_fetch_hints(table, source="unit")
    message = str(refusal.value)
    assert "cadence" in message and "whole number of hours" in message


@pytest.mark.parametrize("source,hours,cadence", [
    ("gdas", 9.0, 2),          # the silently-truncated window, reached
    ("gdas", "9", 2),          # through the OTHER operand of one check
    ("gfs", "7", 3),
    ("gfs", 7.5, 3),
    ("era5", "6", 6),          # the source whose planner reads the key
    ("era5", 6.0, 6),          # first, and read it before the check ran
])
def test_r_an_hours_spelling_cannot_skip_the_window_check(
        source, hours, cadence):
    """The window check has two operands and the fix covered one.

    A decimal or quoted ``hours`` passes the scalar check and then skips
    the planner exactly as a decimal cadence did, so the pair that the
    cadence spelling closed stayed open for the length: `hours = 9.0`
    with `cadence = 2` on GDAS loaded, and it is the silently-truncating
    window the cadence divisibility check exists to prevent.

    The ERA5 rows are the half a per-source ordering can lose: its own
    valid-time planner reads the key inside a branch that ran BEFORE the
    spelling check, so the same two spellings left the door as a
    traceback out of a private helper instead of a refusal naming the
    key and the way out.
    """

    table = {"source": source, "cycle": "2026-07-29T00", "hours": hours,
             "cadence": cadence}
    with pytest.raises(ValueError) as refusal:
        fetch.validate_fetch_hints(table, source="unit")
    message = str(refusal.value)
    assert "hours" in message and "whole number of hours" in message
    # The remedy names the number meant, not just the rule (rule 1).
    assert "what to do" in message


@pytest.mark.parametrize("hours", ["6", 6.0])
def test_r_no_source_branch_reads_the_window_before_it_is_checked(hours):
    """The ordering claim itself, asserted of every front door source.

    A per-source branch that reads ``hours`` ahead of the spelling check
    turns a badly spelled length into whatever its own planner raises.
    Naming the sources that have such a branch would only re-state the
    implementation, so this asks every source a ``[fetch]`` table may
    name and admits exactly one answer: a ValueError naming the key.
    """

    sources = fetch.fetch_front_door_sources()
    assert sources, "no front door source; the ordering has no subject"
    for source in sources:
        table = {"source": source, "cycle": "2026-07-29T00", "hours": hours}
        with pytest.raises(ValueError) as refusal:
            fetch.validate_fetch_hints(table, source="unit")
        message = str(refusal.value)
        assert "hours" in message and "what to do" in message, source


def test_r_single_analysis_remains_valid_acquisition():
    for source in fetch.fetch_front_door_sources():
        fetch.validate_fetch_hints({"source": source, "cycle": _archive_test_cycle(source), "hours": 0},
                                  source="unit")


def test_r_a_whole_cadence_still_plans_its_window():
    fetch.validate_fetch_hints(
        {"source": "gdas", "cycle": "2026-07-29T00", "hours": 6, "cadence": 3},
        source="unit")


# ---------------------------------------------------------------------------
# One emission, one fetch: the front door may not write a flag its own
# fetch refuses
# ---------------------------------------------------------------------------

class _Contacted(Exception):
    """The transport was reached, so plan review accepted the argv."""


def _no_transport(monkeypatch):
    """Make every network primitive raise, so only plan review can run."""

    import socket

    from gpuwm import fetch_pool

    def contact(*_args, **_kwargs):
        raise _Contacted("transport reached")

    for module in (fetch, fetch_routes, fetch_pool):
        for name in ("urlopen", "paced_urlopen"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, contact)
    monkeypatch.setattr(socket, "socket", contact)
    monkeypatch.setattr(fetch, "require_published_cycle", contact)
    monkeypatch.setattr(fetch, "resolve_latest_cycle", contact)
    from gpuwm import era5_acquisition, era5_arco
    monkeypatch.setattr(era5_acquisition, "_client", contact)
    monkeypatch.setattr(era5_arco, "retrieve_era5_arco", contact)


def _archive_test_cycle(source):
    from gpuwm import cf_archive_fetch
    if fetch.native_cf_fetch_contract(source) is not None:
        return cf_archive_fetch.row(fetch_routes.canonical_source(source))["coverage_start"][:13]
    return "2026-08-17T00"


def _emit_domain(tmp_path, source):
    """`gpuwm domain` for SOURCE, at a point inside its own coverage."""

    from gpuwm.source_adapters import source_coverage_window
    from gpuwm.source_coverage import window_centre

    centre = window_centre(source_coverage_window(source)) or (38.5, -97.5)
    out = tmp_path / (source.replace("-", "_") + ".toml")
    rc = cli.main([
        "domain", f"--point={centre[0]:.4f},{centre[1]:.4f}",
        "--card", "16gb", "--root-dx", "3", "--hours", "6",
        "--source", source, "--cycle", _archive_test_cycle(source), "--out", str(out)])
    assert rc == 0, f"{source} did not plan"
    return out


def _planable_sources():
    from gpuwm.source_adapters import wizard_planable_source_ids

    return wizard_planable_source_ids()


@pytest.mark.parametrize("source", _planable_sources())
def test_r_a_wizard_emission_is_never_refused_by_its_own_fetch(
        tmp_path, monkeypatch, capsys, source):
    """What `gpuwm domain` writes, `gpuwm fetch` must accept.

    The emitted ``[fetch]`` table becomes an argv at stage 1 of every
    `gpuwm go`, and a key the fetch door refuses turns a configuration
    that passed `gpuwm check` into a run that dies at its own first
    stage -- a refusal after the run started, and two doors holding two
    answers about one configuration.

    Asserted against the REAL doors on both sides and without naming a
    source: every planable row is emitted, converted by the run plan's
    own argv builder, and pushed through `gpuwm fetch`'s plan review
    with the transport cut.  Reaching the transport is acceptance; a
    ``ValueError`` is the fetch door's refusal and the defect.  A source
    prepared from bytes on disk has no download argv to check, and is
    held to the other half of the same rule: it must carry the cadence
    its staging check needs.
    """

    import tomllib

    out = _emit_domain(tmp_path / source, source)
    capsys.readouterr()
    table = tomllib.loads(out.read_text(encoding="utf-8")).get("fetch")
    assert table, f"{source} emitted no [fetch] table"

    if runplan.drivability_for(source).get("requires_source_root"):
        assert "cadence" in table, (
            f"{source} is prepared from input bytes and its staging check "
            "needs the valid-time spacing")
        return

    argv = runplan._fetch_arguments_from_hints(  # noqa: SLF001
        dict(table), out=tmp_path / "out")
    namespace = cli.build_parser().parse_args(["fetch", *argv])
    _no_transport(monkeypatch)
    try:
        fetch.fetch_main(namespace)
    except _Contacted:
        pass                      # plan review passed, bytes were asked for
    except ValueError as refusal:  # the fetch door's own refusal
        pytest.fail(f"{source}: `gpuwm domain` emitted a [fetch] table its "
                    f"own fetch refuses: {refusal}")
    except RuntimeError:
        pass                      # publication probe, already past review
    finally:
        capsys.readouterr()


def test_r_the_cadence_key_and_the_cadence_flag_have_one_answer():
    """Both doors ask one function whether a cadence applies at all.

    The flag door carried the answer as a sentence inside one source's
    dispatch branch, so the table door had no opinion and the front door
    that writes tables held a third copy.
    """

    from gpuwm.source_adapters import wizard_planable_source_ids

    refusing = [source for source in wizard_planable_source_ids()
                if fetch_routes.source_adapters.get_source_adapter(source).fetch_entire_window]
    assert refusing, "no source refuses a cadence; the seam has no subject"
    for source in refusing:
        native = int(fetch_routes.source_adapters.get_source_adapter(source).forcing_interval_seconds / 3600)
        wrong = native + 1
        # The table door.
        with pytest.raises(ValueError) as table_refusal:
            fetch.validate_fetch_hints(
                {"source": source, "cycle": "2026-07-29T00", "hours": 6,
                 "cadence": wrong}, source="unit")
        # The flag door, asked through the real parser.
        namespace = cli.build_parser().parse_args(
            ["fetch", "--source", source, "--cycle", "2026-07-29T00",
             "--hours", "6", "--cadence", str(wrong), "--out", "d"])
        with pytest.raises(ValueError) as flag_refusal:
            fetch.fetch_main(namespace)
        for message in (str(table_refusal.value), str(flag_refusal.value)):
            assert "cadence" in message
            assert "What to do" in message or "what to do" in message
        assert (fetch.cadence_inapplicable_refusal(source)
                in str(flag_refusal.value))


def test_r_a_cadence_free_source_still_fetches_without_one(
        tmp_path, monkeypatch):
    """Retiring the branch may not retire the acceptance beside it.

    Hoisting the refusal out of one source's dispatch branch moves it in
    front of every source's review, so the half that must NOT refuse is
    asserted at the same door: the argv with no cadence in it reaches the
    transport.  An argparse default is not that assertion -- it holds
    while the acceptance path is broken outright.
    """

    from gpuwm.source_adapters import wizard_planable_source_ids

    cadence_free = [source for source in wizard_planable_source_ids()
                    if fetch_routes.source_adapters.get_source_adapter(source).fetch_entire_window
                    and not runplan.drivability_for(source).get(
                        "requires_source_root")]
    assert cadence_free, "no downloadable source refuses a cadence"
    for source in cadence_free:
        namespace = cli.build_parser().parse_args(
            ["fetch", "--source", source, "--cycle", "2026-07-29T00",
             "--hours", "6", "--out", str(tmp_path / source)])
        assert namespace.cadence is None
        _no_transport(monkeypatch)
        try:
            fetch.fetch_main(namespace)
        except _Contacted:
            pass                  # plan review passed, bytes were asked for
        except ValueError as refusal:
            pytest.fail(f"{source}: the fetch with no cadence in it was "
                        f"refused: {refusal}")
        except RuntimeError:
            pass                  # publication probe, already past review


def test_r_the_local_input_verdict_has_one_reader(monkeypatch):
    """The table door reads the verdict, it does not hold a second copy.

    A configuration naming a source prepared from input bytes on disk is
    admitted by one fact, and two spellings of that fact admit different
    rows the day a registry row makes them differ: the table door built
    its own from the adapter, the runner table and the composition
    document, while review asked the drivability verdict, which requires
    two further facts of the row.  Moving the verdict moves this door.
    """

    from gpuwm import source_drivability
    from gpuwm.source_adapters import (source_forcing_interval_seconds,
                                       wizard_planable_source_ids)

    local = [source for source in wizard_planable_source_ids()
             if runplan.drivability_for(source).get("requires_source_root")]
    assert local, "no local-input source; the admission has no subject"
    source = local[0]
    # The row's own spacing: the subject here is the admission, and a
    # cadence the source's preparation does not take is refused first.
    cadence = int(source_forcing_interval_seconds(source) // 3600)
    table = {"source": source, "cycle": "2026-07-29T00", "hours": 6 * cadence,
             "cadence": cadence}
    fetch.validate_fetch_hints(dict(table), source="unit")

    verdicts = source_drivability.drivability_for

    def without_the_admission(name):
        verdict = dict(verdicts(name))
        verdict.pop("requires_source_root", None)
        return verdict

    monkeypatch.setattr(source_drivability, "drivability_for", without_the_admission)
    # Review reads the same verdict, so both doors move together.
    assert not runplan.drivability_for(source).get("requires_source_root")
    with pytest.raises(ValueError) as refusal:
        fetch.validate_fetch_hints(dict(table), source="unit")
    assert str(table["source"]) in str(refusal.value)


def test_r_a_cadence_the_product_is_not_published_on_is_layered():
    """The refusal argparse used to give now has the layered form.

    ``--cadence`` carried a ``choices`` tuple that answered for every
    source, which is why an any-whole-hour source could not be handed its
    own spacing.  Removing it made the window planner's own sentence
    reachable at the door, and a refusal a reader can reach has to name
    the action and keep the mechanism for ``--explain``.
    """

    import re

    from gpuwm import explain

    # f124 is absent from the published ladder. Four-hour spacing within
    # the hourly part is valid; the requested leads decide the answer.
    with pytest.raises(ValueError) as refusal:
        fetch.gfs_forecast_hours(8, 4, 120)
    action, why = explain.split(str(refusal.value))
    assert why, "the refusal kept no mechanism half for --explain"
    assert "What to do" in action and "Why:" in why
    assert "Why:" not in action
    # And the remedy is checked against the planner rather than read:
    # every spacing IT offers has to plan a window of its own.  Read from
    # the remedy alone, because the sentence above it names the spacing
    # that was refused.
    remedy = action.split("What to do:", 1)[1]
    offered = {int(step) for step in re.findall(r"--cadence (\d+)", remedy)}
    assert offered, "the refusal names no spacing to use instead"
    for step in sorted(offered):
        assert fetch.gfs_forecast_hours(step * 2, step)


# ---------------------------------------------------------------------------
# The predicate and the writer ask one question
# ---------------------------------------------------------------------------

def test_r_the_handoff_writer_forks_on_the_capability_not_the_name(
        tmp_path, monkeypatch):
    """`publishes_prep_handoff` promised a document the writer skipped.

    The predicate answers from the packaged composition; the writer
    branched on the container's name.  Give the other container the same
    composed profile and nothing but the name differs: the predicate says
    a handoff is published and the writer has to publish one.
    """

    real = fetch_routes.source_adapters.get_source_adapter
    composed = real("gdas").packaged_profile

    def adapter_for(name):
        adapter = real(name)
        if adapter.source_id == "gfs":
            return dataclasses.replace(adapter, packaged_profile=composed)
        return adapter

    monkeypatch.setattr(fetch_routes.source_adapters, "get_source_adapter",
                        adapter_for)
    assert fetch_routes.publishes_prep_handoff("gfs") is True
    _legacy_handoff(tmp_path, source="gfs")
    handoff = tmp_path / fetch_routes.PREP_ARGUMENTS_NAME
    assert handoff.is_file()
    document = json.loads(handoff.read_text(encoding="utf-8"))
    assert document["source"] == "gfs"
    role = fetch_routes.in_band_supplement_role("gfs")
    assert any(str(token).startswith(f"{role}=") for token in document["argv"])


def test_r_a_composed_container_that_binds_no_role_still_refuses(
        tmp_path, monkeypatch):
    """Retiring the name branch may not retire the refusal behind it."""

    monkeypatch.setattr(fetch_routes, "in_band_supplement_role",
                        lambda source: None)
    with pytest.raises(ValueError, match="in-band surface binding"):
        _legacy_handoff(tmp_path, source="gdas")


def test_r_an_out_on_a_local_input_config_is_refused_by_name(tmp_path):
    """The fourth accept-and-ignore key in the same table.

    ``area``, ``point`` and ``radius_km`` were refused by name on a
    local-input configuration and ``out`` -- which names where a download
    would write its files -- was left beside them, read and applied to
    nothing.  The front door stops emitting it for such a source, so the
    emission's own round-trip proves the pair.
    """

    local = [source for source in _planable_sources()
             if runplan.drivability_for(source).get("requires_source_root")]
    assert local, "no local-input source; the refusal has no subject"
    for source in local:
        with pytest.raises(ValueError) as refusal:
            fetch.validate_fetch_hints(
                {"source": source, "cycle": "2026-07-29T00", "hours": 6,
                 "cadence": 3, "out": str(tmp_path)}, source="unit")
        message = str(refusal.value)
        assert "out" in message and "source_root" in message
        assert "what to do" in message


@pytest.mark.parametrize("hours,cadence", [(10, 4), (7, 3)])
def test_r_an_era5_window_the_retrieval_refuses_is_refused_at_load(
        hours, cadence):
    """ERA5 gained cadences, and with them accepted-then-refused windows.

    The retrieval plans its valid times with one function; the table door
    checked the product clock and never the window, so a length the
    cadence does not divide loaded clean and died at the fetch.
    """

    table = {"source": "era5", "cycle": "2026-07-29T00", "hours": hours,
             "cadence": cadence, "area": "30,-110,45,-85"}
    with pytest.raises(ValueError) as refusal:
        fetch.validate_fetch_hints(table, source="unit")
    assert "multiple of the" in str(refusal.value)
    # A window the cadence does divide still loads.
    fetch.validate_fetch_hints({**table, "hours": cadence * 2}, source="unit")


def test_r_the_fetch_door_forks_on_the_capability_not_the_name(
        tmp_path, monkeypatch):
    """The plan-review check and the writer ask one question.

    The writer forks on whether a packaged composition drives this
    preparation; the fetch door's plan-review call was gated on a
    container's NAME, so a container that gains a composed profile had
    its review skipped while the writer still demanded a binding after
    the download -- the refusal moving back behind the transfer.
    """

    real = fetch_routes.source_adapters.get_source_adapter
    composed = real("gdas").packaged_profile

    def adapter_for(name):
        adapter = real(name)
        if adapter.source_id == "gfs":
            return dataclasses.replace(adapter, packaged_profile=composed)
        return adapter

    monkeypatch.setattr(fetch_routes.source_adapters, "get_source_adapter",
                        adapter_for)
    monkeypatch.setattr(fetch_routes, "in_band_supplement_role",
                        lambda source: None)
    assert fetch_routes.prepares_through_packaged_composition("gfs") is True

    def never(*_args, **_kwargs):
        raise AssertionError("the transport was reached before the refusal")

    monkeypatch.setattr(fetch, "require_published_cycle", never)
    monkeypatch.setattr(fetch, "fetch_gfs", never)
    namespace = cli.build_parser().parse_args([
        "fetch", "--source", "gfs", "--cycle", "2026-07-29T00",
        "--hours", "6", "--cadence", "3", "--area", "30,-110,45,-85",
        "--out", str(tmp_path)])
    with pytest.raises(ValueError, match="in-band surface binding"):
        fetch.fetch_main(namespace)


def test_r_the_composition_document_has_one_reader():
    """Caching a read that a second module still makes is not caching it.

    The acquisition modules ask
    :func:`gpuwm.source_authorities.packaged_composition`, which verifies
    and parses once per file generation.  A module that opens the same
    document itself re-reads and re-hashes it on every call, which is the
    cost the cache was added to remove.
    """

    import ast
    import gpuwm

    root = Path(gpuwm.__file__).parent
    second_readers = []
    for name in ("local_preparation.py", "prep_handoff.py",
                 "fetch_routes.py", "runplan.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            index = node.slice
            if isinstance(index, ast.Constant) and index.value == "composition":
                second_readers.append(f"{name}:{node.lineno}")
    assert second_readers == [], (
        "these lines index a profile's composition file themselves instead "
        "of asking source_authorities.packaged_composition")
