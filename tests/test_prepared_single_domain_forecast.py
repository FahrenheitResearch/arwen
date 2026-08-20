from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.experiment import load_experiment
from gpuwm.ingest.prepared_cache import (
    PREPARED_CACHE_SCHEMA,
    _array_sha256,
    prepared_cache_identity,
)
from gpuwm.ingest.hrrr_physics import (
    _validate_prepared_near_surface,
    _validate_prepared_surface,
)
import tools.prepared_single_domain_forecast as runner


ROOT = Path(__file__).parents[1]

_TOML_HEADER = re.compile(
    r"^\s*(\[\[|\[)([A-Za-z0-9_.-]+)(\]\]|\])\s*(?:#.*)?$")
_TOML_ASSIGNMENT = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*=")


def _asking_for_no_physics(text: str) -> str:
    """The same case with every profile-owned physics declaration removed.

    CONTRACT CHANGE 2026-08-09 (ledger #90).  ``--materialize-authorities``
    used to DELETE all 26 profile-owned keys from whatever config it was
    handed and rewrite them from the named profile, silently; the fixtures
    below leaned on that to turn one shipped WSM6 proof config into a base
    for any suite.  It now SUPPLIES the profile-owned keys a config is
    silent about and REFUSES, by name, the ones the config states
    differently -- so a fixture that wants to see what profile P produces
    hands it a base that has asked for nothing, which is exactly the
    situation the supply half of the contract exists for.

    The materialized bytes are unchanged by the repair: every one of the
    1872 shipped-config x source x profile pairs renders byte-identically
    from the stripped base to what the old stripping renderer produced
    from the declared one, so every digest and every switch these tests
    pin is the same value it was.
    """

    kept: list[str] = []
    section: str | None = None
    for line in text.splitlines():
        header = _TOML_HEADER.match(line)
        if header is not None:
            section = header.group(2)
            kept.append(line)
            continue
        assignment = _TOML_ASSIGNMENT.match(line)
        if (section in {"shared", "domain"} and assignment is not None
                and assignment.group(1) in runner._MATERIALIZED_PHYSICS_KEYS):
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def _no_physics_copy(tmp_path: Path, config: Path, *,
                     name: str | None = None) -> Path:
    """``config`` with its physics declarations dropped, on disk."""

    copy = tmp_path / (name or f"no-physics-{config.name}")
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(
        _asking_for_no_physics(config.read_text(encoding="utf-8")),
        encoding="utf-8")
    return copy


def _declaring_before_first_domain(text: str, block: str) -> str:
    """``text`` with ``block`` inserted ahead of the first REAL [[domain]].

    Anchored on a header LINE rather than the first occurrence of the
    substring: several shipped configs mention ``[[domain]]`` inside a
    comment first (configs/real74_2dom_gf.toml at char 872, for one),
    and a plain ``str.replace`` there splices TOML into prose and the
    fixture stops parsing.  The verifier's sweep hit exactly that.
    """

    match = re.search(r"^\[\[domain\]\]", text, re.MULTILINE)
    assert match is not None, "fixture config has no [[domain]] table"
    return text[:match.start()] + block + text[match.start():]


def _mapped_proof_literals() -> dict[str, set[str]]:
    """The top-level keys ``gpuwm/mapped_direct.py`` actually writes.

    Read out of the writer's own source with :mod:`ast`, from the two
    ``proof = {...}`` assignments it publishes -- one direct, one
    hierarchy, told apart by the schema constant each names.  The one
    computed key, ``forcing_key``, is the forcing axis, and it is
    normalized to ``forcing_hours`` because that is the only spelling
    the reader accepts (a sub-hourly mapped preparation is refused
    elsewhere, by name).

    This exists because reading a writer's source is still a great deal
    better than restating its inventory in the reader and hoping.  When
    the reader restated it, the 2.5.0 soil-mesh work added
    ``soil_texture_downscale`` to both proofs and the forecast runner
    silently began rejecting every mapped bundle the preparation stage
    produced.  Nothing failed, because the fixture below wrote its own
    proof rather than the writer's.
    """

    import ast

    source = (ROOT / "gpuwm" / "mapped_direct.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(
                node.value, ast.Dict):
            continue
        if not (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "proof"):
            continue
        keys: set[str] = set()
        schema_name = None
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
                if key.value == "schema" and isinstance(value, ast.Name):
                    schema_name = value.id
            elif isinstance(key, ast.Name) and key.id == "forcing_key":
                keys.add("forcing_hours")
            else:  # pragma: no cover - a new computed key needs a decision
                raise AssertionError(
                    "gpuwm/mapped_direct.py grew a proof key this gate "
                    "cannot read statically; teach it, do not delete it")
        # `proof_content_sha256` is assigned on the next statement, not
        # inside the literal, and both documents carry it.
        keys.add("proof_content_sha256")
        assert schema_name is not None
        found[schema_name] = keys
    return found


def test_the_reader_and_the_writer_agree_on_the_mapped_proof_inventory():
    """THE gate for the drift that killed the mapped route on this line.

    ``gpuwm.mapped_direct`` writes the mapped/20CRv3 preparation proof;
    ``prepared_single_domain_forecast`` refuses any proof whose exact
    top-level inventory it does not recognise.  Those two lists have to
    be the same list.  When they were not, a freshly prepared bundle was
    refused with "mapped 20CRv3 proof top-level inventory differs" --
    the last leg of the whole mapped/arbitrary-source chain, dead, with
    every test green.
    """

    literals = _mapped_proof_literals()
    assert set(literals) == {"PROOF_SCHEMA", "HIERARCHY_PROOF_SCHEMA"}, (
        "gpuwm/mapped_direct.py no longer publishes exactly two proof "
        f"documents; it publishes {sorted(literals)}")
    assert literals["PROOF_SCHEMA"] == set(runner.MAPPED_DIRECT_PROOF_KEYS), (
        "the mapped direct proof the writer publishes and the inventory "
        "the forecast runner accepts have drifted apart")
    assert literals["HIERARCHY_PROOF_SCHEMA"] \
        == set(runner.MAPPED_HIERARCHY_PROOF_KEYS), (
        "the mapped hierarchy proof the writer publishes and the "
        "inventory the forecast runner accepts have drifted apart")


def test_prepared_runner_capability_query_is_side_effect_free_without_run_args(
        tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert runner.main(["--show-capabilities"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == runner.runner_capabilities()
    assert payload["schema"] == "gpuwm-runner-capabilities-v1"
    assert payload["supported_sources"] == [
        "20crv3", "20crv3-cf", "aifs", "aigefs", "aigfs",
        "ecmwf-open-data", "era5", "era5-l137", "gdas", "gefs",
        "gem-gdps", "gfs", "hrrr", "hrrr-prs", "icon-eu", "rap", "rrfs"]
    assert payload["physics_profile_ids"] == list(runner.PHYSICS_PROFILES)
    assert payload["report_schema"] == runner.REPORT_SCHEMA
    assert payload["window"]["limit_policy"] \
        == "prepared-cache-forcing-coverage"
    assert payload["window"]["maximum_run_seconds"] is None
    assert payload["window"]["run_seconds"]["whole_hour_required"] is True
    assert payload["window"]["source_forcing_cadence_hours"] == {
        "gfs": [1, 3],
        "era5": "uniform-positive-whole-hour",
        "20crv3": "manifest-bound-uniform-positive-whole-hour",
        "20crv3-cf": "uniform-positive-whole-hour",
        "hrrr-prs": "uniform-positive-whole-hour",
        "aifs": "uniform-positive-whole-hour",
        "aigefs": "uniform-positive-whole-hour",
        "aigfs": "uniform-positive-whole-hour",
        "ecmwf-open-data": "uniform-positive-whole-hour",
        "gdas": "uniform-positive-whole-hour",
        "gefs": "uniform-positive-whole-hour",
        "gem-gdps": "uniform-positive-whole-hour",
        "icon-eu": "uniform-positive-whole-hour",
        "rap": "uniform-positive-whole-hour",
        "rrfs": "uniform-positive-whole-hour",
    }
    twentycr = payload["source_profiles"]["20crv3"]
    assert twentycr["member_identity"] \
        == "filename_memNNN_not_grib2_pdt"
    assert "NOT_ACCEPTANCE_GATED" in twentycr["readiness"]
    # The packaged NetCDF profile is a source of its own here, and the two
    # things a reader must not learn late are stated in its limitations.
    twentycr_cf = payload["source_profiles"]["20crv3-cf"]
    assert "NOT_ACCEPTANCE_GATED" in twentycr_cf["readiness"]
    assert twentycr_cf["member_identity"]         == "ensemble_mean_analysis_not_a_member"
    assert any("ENSEMBLE MEAN" in limit
               for limit in twentycr_cf["limitations"])
    assert any("orography and land mask are recovered" in limit
               for limit in twentycr_cf["limitations"])
    assert payload["output"]["io_modes"] == ["history"]
    assert payload["output"]["configurable_cadence"] is True
    cadence = payload["output"]["history_interval_seconds"]
    assert cadence["must_equal_hash_bound_experiment"] is True
    assert cadence["must_be_whole_model_steps"] is True
    assert cadence["must_evenly_divide_run"] is False
    assert cadence["last_scheduled_frame_may_precede_run_end"] is True
    assert payload["readiness"] \
        == "FORECAST_IMPLEMENTATION_PRESENT_RUNTIME_PREFLIGHT_REQUIRED"
    assert payload["modes"]["forecast"]["available"] is True
    assert payload["standalone_rw_wps_wheel"]["runner_included"] is False
    assert payload["standalone_rw_wps_wheel"][
        "forecast_executor_included"] is False
    thompson = payload["physics_profiles"][runner.THOMPSON_PHYSICS_PROFILE]
    assert thompson["explicit_expert_consent_required"] is False
    assert thompson["table_root"]["environment"] \
        == runner.THOMPSON_TABLE_ROOT_ENV
    assert len(thompson["table_authority"]["assets"]) == 4
    nssl2 = payload["physics_profiles"][runner.NSSL2_PHYSICS_PROFILE]
    assert nssl2["readiness"] == "WRF_MATCHED_RUN_CANDIDATE"
    assert nssl2["explicit_expert_consent_required"] is False
    assert nssl2["resolved_fixed_preset"] is True
    legacy = payload["physics_profiles"][
        runner.NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE]
    assert legacy["readiness"] == "WRF_MATCHED_RUN_CANDIDATE"
    assert legacy["radiation_solver"] == "legacy RRTMG"
    twentycr_wsm6 = payload["physics_profiles"][
        runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE]
    assert twentycr_wsm6["readiness"] == "IMPLEMENTED_UNVERIFIED"
    assert twentycr_wsm6["source_scope"] == ["20crv3"]
    assert payload["capability_query"]["requires_cupy"] is False
    materializer = payload["authority_materialization"]
    assert materializer["available"] is True
    assert materializer["mode_flag"] == "--materialize-authorities"
    assert materializer["receipt_schema"] \
        == runner.AUTHORITY_MATERIALIZATION_SCHEMA
    assert runner.MORRISON_PHYSICS_PROFILE in materializer[
        "source_physics_profile_ids"]["gfs"]
    assert materializer["source_physics_profile_ids"]["20crv3"][0] \
        == runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE
    assert list(tmp_path.iterdir()) == []


def test_prepared_capabilities_fail_closed_when_executor_modules_are_absent(
        monkeypatch):
    monkeypatch.setattr(
        runner,
        "_missing_forecast_executor_modules",
        lambda: ["gpuwm.core.model"],
    )

    payload = runner.runner_capabilities()

    assert payload["readiness"] == "FORECAST_EXECUTOR_OMITTED"
    assert payload["modes"]["forecast"]["available"] is False
    assert payload["modes"]["forecast"]["missing_executor_modules"] == [
        "gpuwm.core.model"]
    assert payload["modes"]["forecast"]["unavailable_reason"] \
        == "GPUWM forecast executor modules are absent"
    assert payload["supported_sources"] == []
    assert payload["physics_profile_ids"] == []
    assert payload["source_profiles"] == {}
    assert payload["physics_profiles"] == {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False)


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")


def _artifact(path: Path, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bind_test_thompson_runtime(tmp_path: Path, monkeypatch):
    root = tmp_path / "thompson-tables"
    root.mkdir()
    asset_type = type(runner.THOMPSON_CLASSIC_TABLE_ASSETS[0])
    assets = []
    for canonical in runner.THOMPSON_CLASSIC_TABLE_ASSETS:
        payload = f"test-{canonical.filename}".encode("ascii")
        path = root / canonical.filename
        path.write_bytes(payload)
        assets.append(asset_type(
            canonical.filename, len(payload), hashlib.sha256(payload).hexdigest()))
    assets = tuple(assets)

    def validate(path):
        assert Path(path).resolve() == root.resolve()
        for asset in assets:
            current = root / asset.filename
            if current.stat().st_size != asset.bytes \
                    or _sha256(current) != asset.sha256:
                raise ValueError(f"test Thompson table drift: {asset.filename}")
        return assets

    # The enable gate is retired; the table root stays overridable so a
    # test can point the byte validation at its own fixture set.
    monkeypatch.setenv(runner.THOMPSON_TABLE_ROOT_ENV, str(root))
    monkeypatch.setattr(runner, "THOMPSON_CLASSIC_TABLE_ASSETS", assets)
    monkeypatch.setattr(runner, "validate_thompson_table_assets", validate)
    return SimpleNamespace(root=root.resolve(), assets=assets)


@pytest.mark.parametrize(
    ("source", "profile"),
    (
        ("gfs", runner.PHYSICS_PROFILE),
        ("gfs", runner.THOMPSON_PHYSICS_PROFILE),
        ("era5", runner.MORRISON_PHYSICS_PROFILE),
        ("era5", runner.NSSL2_PHYSICS_PROFILE),
        ("20crv3", runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE),
        ("20crv3", runner.MORRISON_PHYSICS_PROFILE),
    ),
)
def test_materializer_supplies_the_profile_a_silent_config_asked_nothing_of(
        tmp_path, monkeypatch, source, profile):
    """THE PROMISE, rewritten 2026-08-09 (ledger #90).

    OLD PROMISE, which this test used to encode: hand
    ``--materialize-authorities`` any config at all and the named
    profile's 26 switches come out the other side, because the
    materializer DELETED whatever physics the config declared and
    rewrote it.  The base here was the shipped WSM6 proof config
    (``mp_physics = 6``, no longwave, no cumulus) and the assertion was
    that a Morrison authority came out of it anyway, silently.  That is
    the owner's "never silently replace a physics value the user wrote"
    rule broken at step 2 of the documented six.

    NEW PROMISE: a named profile SUPPLIES every profile-owned key the
    config is silent about, exactly and completely -- which is what this
    test now measures, on a base that declares no physics at all.  The
    other half of the contract, that a key the config DOES state is
    refused rather than overwritten, is
    ``test_materializer_refuses_to_overwrite_a_declared_physics_value``
    below.  The generated bytes are unchanged by the split: the switches,
    digests and receipts asserted here are the same values the old
    stripping renderer produced.
    """

    if profile == runner.THOMPSON_PHYSICS_PROFILE:
        _bind_test_thompson_runtime(tmp_path, monkeypatch)
    config_source = "gfs" if source in {"gfs", "20crv3"} else "era5"
    base_experiment = _no_physics_copy(
        tmp_path,
        ROOT / "configs" / f"{config_source}_wrf_direct_proof.toml")
    base_wps = (
        ROOT / "configs" / f"{config_source}_wrf_direct_proof.namelist.wps")
    # The base really is silent: nothing below is a value being replaced.
    base_text = base_experiment.read_text(encoding="utf-8")
    assert not [
        key for key in runner._MATERIALIZED_PHYSICS_KEYS
        if re.search(rf"^\s*{key}\s*=", base_text, re.MULTILINE)]
    base = load_experiment(base_experiment)
    output = tmp_path / f"{source}-{profile[:4]}"

    receipt = runner.materialize_named_source_authorities(
        source=source,
        base_experiment_config=base_experiment,
        base_wps_namelist=base_wps,
        physics_profile=profile,
        output_directory=output,
    )

    generated = load_experiment(output / "experiment.toml")
    expected = runner._profile_runtime_switches(source, profile)
    assert receipt["schema"] == runner.AUTHORITY_MATERIALIZATION_SCHEMA
    assert receipt["status"] == "PASS"
    assert receipt["normalized_selected_physics"]["resolved"] == {
        **expected,
        "radiation_scheme_ids": list(
            runner.radiation_scheme_ids(generated.root.run)),
    }
    assert receipt["non_physics_descriptor"]["status"] == "EXACT_UNCHANGED"
    assert receipt["non_physics_descriptor"]["base_sha256"] \
        == receipt["non_physics_descriptor"]["generated_sha256"]
    assert generated.start_time == base.start_time
    assert generated.run_seconds == base.run_seconds
    assert generated.vertical == base.vertical
    assert generated.projection == base.projection
    assert [domain.grid_id for domain in generated.domains] \
        == [domain.grid_id for domain in base.domains]
    # Supplied exactly and completely, switch for switch, on every domain.
    for domain in generated.domains:
        assert {name: getattr(domain.run, name) for name in expected} \
            == expected, domain.grid_id
    assert (output / "namelist.wps").read_bytes() == base_wps.read_bytes()
    persisted = json.loads(
        (output / "authority-receipt.json").read_text(encoding="utf-8"))
    assert persisted["generated"]["experiment_config"]["sha256"] \
        == _sha256(output / "experiment.toml")
    assert receipt["receipt"]["sha256"] \
        == _sha256(output / "authority-receipt.json")


def test_materializer_refuses_to_overwrite_a_declared_physics_value(tmp_path):
    """The other half of the 2026-08-09 contract: no silent replacement.

    The shipped GFS proof config declares a complete WSM6 suite.  Naming
    Morrison over it used to delete all seven disagreeing values and
    publish a Morrison authority, with nothing said and nothing in the
    receipt; the run that followed was not the run the config described.
    It refuses now, at step 2 of the documented six -- before the fetch,
    before rw-wps -- naming every key, both values and both remedies,
    and creating nothing.

    THE DISAGREEMENT SET MOVED (1.8.8).  Through 1.8.7 this config ran
    Dudhia shortwave with longwave OFF, so the set it contradicted the
    Morrison profile on included ``ra_lw_physics``, ``ra_sw_physics``
    and ``radt``.  The GLW lane re-specified it to a real 4/4 legacy
    RRTMG suite, which is the right config change and leaves those three
    keys in agreement -- so this test now pins the SIX keys that still
    disagree.  The radiation half of the coverage did not go away with
    them: it moved to the sibling below, which builds a config that
    contradicts the profile's radiation on purpose, because a test that
    silently stops checking radiation is worse than one that fails.
    """

    declared = ROOT / "configs" / "gfs_wrf_direct_proof.toml"
    output = tmp_path / "authority"

    with pytest.raises(ValueError) as caught:
        runner.materialize_named_source_authorities(
            source="gfs", base_experiment_config=declared,
            base_wps_namelist=(
                ROOT / "configs" / "gfs_wrf_direct_proof.namelist.wps"),
            physics_profile=runner.MORRISON_PHYSICS_PROFILE,
            output_directory=output)
    message = str(caught.value)

    # Every disagreeing key by its own name, with the config's value and
    # the profile's, so the reader never has to go and diff two files.
    assert "[shared] mp_physics: config 6, profile 10" in message
    assert ("[shared] ra_rrtmg_variant: config 'rrtmg_legacy', profile "
            "'rte-rrtmgp'") in message
    assert ("[shared] wrf_rrtmg_compatibility: config "
            "'wrf-rrtmg-4-4-legacy-v1', profile "
            "'wrf-rrtmg-4-4-to-rte-rrtmgp-v2'") in message
    assert "[[domain]] d01 cu_physics: config 0, profile 1" in message
    assert "[[domain]] d01 cudt_minutes: config 0.0, profile 5.0" in message
    assert "[[domain]] d01 diff_6th_factor: config 0.08, profile 0.12" \
        in message
    assert "6 values would be overwritten" in message
    # The keys this config now AGREES with the profile on are absent --
    # the same both-directions check the tail of this test makes for
    # keys the config never mentioned.
    assert "ra_lw_physics" not in message
    assert "radt" not in message
    # The offending flag, the offending file, and BOTH remedies.  The
    # 1.8.8 suite matches no shipped profile, so the second remedy is
    # the omission branch; the sibling below holds the branch that names
    # the config's own suite.
    assert f"--physics-profile {runner.MORRISON_PHYSICS_PROFILE}" in message
    assert str(declared) in message
    assert "edit those keys" in message
    assert ("omit --physics-profile, which publishes the config's own "
            "suite unchanged") in message
    # A refusal, not a half-published case.
    assert not output.exists()
    # Silent success is impossible in the other direction too: keys the
    # config never mentioned are not listed as conflicts.
    assert "num_soil_layers" not in message
    assert "top_lid" not in message


def test_the_drift_refusal_names_a_radiation_disagreement(tmp_path):
    """Radiation drift is reported, on a config that actually drifts.

    The sibling above used to carry this half, because through 1.8.7 the
    shipped GFS proof config ran Dudhia shortwave with longwave off
    while the Morrison profile runs 4/4.  The GLW lane moved that config
    to a real legacy-RRTMG 4/4 suite -- the right change -- and the
    assertions naming ``ra_lw_physics``, ``ra_sw_physics`` and ``radt``
    went stale with it.  Deleting them would have silently deleted the
    coverage: radiation could stop being enumerated by the refusal
    entirely and every test in this file would still be green.

    So the drifting config is BUILT here rather than borrowed, as the
    shipped one was through 1.8.7, and it carries the whole set the
    sibling can no longer see -- both radiation keys, the radiation
    clock, and the ``matched is not None`` remedy branch that names the
    config's own shipped suite (the 1.8.8 config matches no profile, so
    the sibling now exercises only the other branch).  The 0/1 pairing
    loads only with both radiation acknowledgements in ink, so this
    fixture also proves a declared asymmetric config reaches the
    materializer at all rather than being turned away one door earlier.
    """

    shipped = (ROOT / "configs" / "gfs_wrf_direct_proof.toml").read_text(
        encoding="utf-8")
    drifting = (
        shipped
        .replace("ra_lw_physics = 4", "ra_lw_physics = 0")
        .replace("ra_sw_physics = 4", "ra_sw_physics = 1")
        .replace("radt = 12.0", "radt = 1.0")
        # The RRTMG contract keys require the resolved 4/4 pair, so they
        # leave with the pairing they describe.
        .replace('wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-legacy-v1"\n', "")
        .replace('ra_rrtmg_variant = "rrtmg_legacy"\n', "")
        .replace("[experiment]",
                 '[experiment]\nacknowledgements = ['
                 '"asymmetric-radiation-nocturnal-window-v1", '
                 '"constant-downward-longwave-v1"]', 1))
    config = tmp_path / "radiation_drift.toml"
    config.write_text(drifting, encoding="utf-8")
    wps = tmp_path / "radiation_drift.namelist.wps"
    shutil.copy(ROOT / "configs" / "gfs_wrf_direct_proof.namelist.wps", wps)
    output = tmp_path / "authority"

    with pytest.raises(ValueError) as caught:
        runner.materialize_named_source_authorities(
            source="gfs", base_experiment_config=config,
            base_wps_namelist=wps,
            physics_profile=runner.MORRISON_PHYSICS_PROFILE,
            output_directory=output)
    message = str(caught.value)

    assert "[shared] ra_lw_physics: config 0, profile 4" in message
    assert "[shared] ra_sw_physics: config 1, profile 4" in message
    assert "[[domain]] d01 radt: config 1.0, profile 12.0" in message
    assert "7 values would be overwritten" in message
    # The other remedy branch: this suite IS a shipped profile, so the
    # refusal names it rather than falling back to "omit the flag".
    assert (f"--physics-profile {runner.PHYSICS_PROFILE} (the shipped suite "
            "this config already IS)") in message
    assert not output.exists()


def test_a_config_that_already_is_the_profile_materializes_silently(
        tmp_path, capsys):
    """Agreement is not a conversation: matching keys just work.

    This is the documented route -- `gpuwm domain --physics-profile P`
    writes P's switches, step 2 names P over them -- so the refusal must
    be invisible to it.  The config below states all 24 of the profile's
    keys explicitly and materializes with no warning, no refusal, and
    byte-identical output to the same case declaring nothing.
    """

    profile = runner.MORRISON_PHYSICS_PROFILE
    pinned = runner._profile_pinned_physics(
        runner._profile_runtime_switches("gfs", profile))
    silent = _asking_for_no_physics(
        (ROOT / "configs" / "gfs_wrf_direct_proof.toml").read_text(
            encoding="utf-8"))
    agreeing = _declaring_before_first_domain(
        silent,
        "".join(f"{key} = {runner._toml_literal(value)}\n"
                for key, value in sorted(pinned.items())))
    assert agreeing.count("mp_physics = 10") == 1
    capsys.readouterr()

    from_silence, _exp, _receipt = runner._render_materialized_experiment(
        silent, source="gfs", profile=profile)
    from_agreement, exp, receipt = runner._render_materialized_experiment(
        agreeing, source="gfs", profile=profile)

    assert capsys.readouterr().err == ""
    assert from_agreement == from_silence
    assert receipt["profile_validation"]["profile"] == profile
    assert int(exp.root.run.mp_physics) == 10

    # And the same keys with ONE value moved is the refusal, so the pass
    # above is agreement being detected, not the check being asleep.  The
    # moved value is a perfectly valid radiation cadence: what refuses it
    # is the disagreement, not the number.
    with pytest.raises(ValueError, match=r"radt: config 30\.0, profile 12\.0"):
        runner._render_materialized_experiment(
            agreeing.replace("radt = 12.0", "radt = 30.0", 1),
            source="gfs", profile=profile)


def test_the_aggregate_radiation_spelling_is_agreement_not_drift():
    """ONE case where refusing would be wrong, handled explicitly.

    ``ra_lw_physics = ra_sw_physics = -1`` is documented in
    gpuwm/config.py as preserving the historical ``ra_physics``
    aggregate exactly, and gpuwm/ingest/prepared_cache.py already
    canonicalizes the two spellings to each other for cache identity.
    A config that reaches the profile's two schemes through the
    aggregate has not asked for different physics, so a key-by-key
    comparison would refuse a case that runs bit for bit what the
    profile runs.  The comparison is on the resolved pair.
    """

    silent = _asking_for_no_physics(
        (ROOT / "configs" / "gfs_wrf_direct_proof.toml").read_text(
            encoding="utf-8"))
    # 20CRv3 WSM6 pins the aggregate (ra_physics 4, lw/sw -1); the config
    # below spells the identical selection the explicit way.
    profile = runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE
    pinned = runner._profile_pinned_physics(
        runner._profile_runtime_switches("gfs", profile))
    assert (pinned["ra_physics"], pinned["ra_lw_physics"],
            pinned["ra_sw_physics"]) == (4, -1, -1)
    explicit = _declaring_before_first_domain(
        silent,
        "ra_physics = 0\nra_lw_physics = 4\nra_sw_physics = 4\n\n")

    _rendered, exp, _receipt = runner._render_materialized_experiment(
        explicit, source="gfs", profile=profile)
    assert runner.radiation_scheme_ids(exp.root.run) == (4, 4)

    # A genuinely different pair through the same spelling still refuses.
    with pytest.raises(ValueError, match="ra_lw_physics"):
        runner._render_materialized_experiment(
            _declaring_before_first_domain(
                silent,
                "ra_physics = 0\nra_lw_physics = 0\nra_sw_physics = 1\n\n"),
            source="gfs", profile=profile)


def test_a_key_no_profile_pins_is_kept_rather_than_deleted():
    """``radt_minutes`` belongs to no profile, so nothing may eat it.

    It is inside ``_MATERIALIZED_PHYSICS_KEYS`` because that set is also
    the mask on both sides of the non-physics descriptor digest, and no
    switch table pins it.  The old renderer deleted every member of that
    set; a value the profile is not replacing must survive.

    WHAT KEEPING IT MOVES, measured below rather than left implicit
    (2026-08-09 verifier finding): the kept key changes
    ``prepared_domain_config_identity`` -- the strict-equality document
    the prepared cache compares -- for any user config that declares a
    non-default ``radt_minutes``.  A prepared tree built for such a
    config BEFORE this contract change resolves the old deleted-key
    default and no longer matches; it must be re-prepared once.  No
    shipped config declares the key, and the CHANGELOG discloses the
    breakage.  What keeping it does NOT move is the physics: every
    consumption site reads ``radt`` when ``radt`` is positive, and every
    shipped profile pins a positive ``radt``, so under a named profile
    the kept spelling is physically inert -- which is asserted here, not
    assumed, so a future zero-``radt`` profile fails this line instead
    of quietly making the kept key live.
    """

    silent = _asking_for_no_physics(
        (ROOT / "configs" / "gfs_wrf_direct_proof.toml").read_text(
            encoding="utf-8"))
    with_spelling = _declaring_before_first_domain(
        silent, "radt_minutes = 7.0\n\n")

    rendered, exp, receipt = runner._render_materialized_experiment(
        with_spelling, source="gfs",
        profile=runner.MORRISON_PHYSICS_PROFILE)

    assert "radt_minutes = 7.0" in rendered
    # Still outside the descriptor digest, so keeping it proves nothing
    # about the descriptors either way.
    assert (receipt["base_non_physics_descriptor_sha256"]
            == receipt["generated_non_physics_descriptor_sha256"])
    # Inert under this (and every shipped) profile: the pinned positive
    # radt is what the radiation clock reads.
    assert exp.root.run.radt_minutes == 7.0
    assert exp.root.run.radt > 0.0
    # The DISCLOSED cost: the prepared-cache identity document moves on
    # exactly this key relative to the old delete-and-rewrite renderer.
    from gpuwm.ingest.prepared_cache import prepared_domain_config_identity
    _rendered, silent_exp, _receipt = runner._render_materialized_experiment(
        silent, source="gfs", profile=runner.MORRISON_PHYSICS_PROFILE)
    kept = prepared_domain_config_identity(exp.root.run)
    stripped = prepared_domain_config_identity(silent_exp.root.run)
    assert {key for key in kept if kept[key] != stripped.get(key)} \
        == {"radt_minutes"}
    assert (stripped["radt_minutes"], kept["radt_minutes"]) == (12.0, 7.0)


def _wizard_ladder_config(tmp_path: Path) -> Path:
    """The wizard's own two-rung ladder, bound to its own default profile.

    Not a hand-built stand-in: this is the artifact `gpuwm run-plan`'s
    prepared route hands to stage 1, and the wizard writes its nests
    with DELIBERATE per-domain departures from the root suite --
    ``cu_physics = 0`` below the gray zone and the certified
    ``diff_6th_factor`` ladder -- so the root resolves to the named
    profile while the tree as a whole does not.

    ``radt`` was a third departure until 2.5.0's radt-floor repair; the
    wizard's nests now INHERIT the root's radiation cadence, so it is an
    agreeing key here and the two departures above carry this fixture.
    """

    from gpuwm.cli import main as cli_main

    out = tmp_path / "ladder.toml"
    assert cli_main([
        "domain", "--point=35.3,-97.5", "--card", "24gb",
        "--ladder", "12-3", "--source", "gfs",
        "--cycle", "2026-07-29T18", "--hours", "6",
        "--physics-profile", runner.MORRISON_PHYSICS_PROFILE,
        "--out", str(out)]) == 0
    return out


def test_a_nest_that_departs_from_the_profile_is_refused_by_scope(tmp_path):
    """The d02+ half of the drift refusal (2026-08-09 verifier blocker).

    Every original refusal test was single-domain, and the multi-domain
    path is the one that breaks real configs: the OLD renderer deleted
    nest-level physics and let the nests inherit the profile's [shared]
    block -- measured on the wizard's own --ladder emission, that turned
    ``cu_physics = 0`` at 3 km into cumulus ON, ``radt = 3.0`` into 12.0
    and ``diff_6th_factor = 0.1`` into 0.12, silently.  The refusal must
    therefore see every [[domain]] table, and name the nest scope so the
    reader knows WHICH domain's physics the flag contradicts.
    """

    config = _wizard_ladder_config(tmp_path)
    base = load_experiment(config)
    from gpuwm.physics_compat import identify_single_domain_profile
    # The regression's precondition, asserted so the fixture cannot
    # decay: the ROOT alone really does match the named profile.
    assert identify_single_domain_profile(base.root.run) \
        == runner.MORRISON_PHYSICS_PROFILE

    with pytest.raises(ValueError) as caught:
        runner._render_materialized_experiment(
            config.read_text(encoding="utf-8"), source="gfs",
            profile=runner.MORRISON_PHYSICS_PROFILE, origin=str(config))
    message = str(caught.value)
    assert "[[domain]] d02 cu_physics: config 0, profile 1" in message
    assert "[[domain]] d02 diff_6th_factor: config 0.1, profile 0.12" \
        in message
    # radt used to be the third row here (config 3.0, profile 12.0).
    # 2.5.0's radt-floor repair makes a nest inherit the root's
    # radiation cadence, so d02 now AGREES with the profile at 12.0 --
    # and an agreeing key is exactly what this refusal must not list.
    assert "[[domain]] d02 radt" not in message
    # No [shared] rows: the shared block IS the profile here, and a
    # refusal that listed agreeing keys would be listing noise.
    assert "[shared]" not in message.split("[[explain]]")[0]


def test_the_remedy_for_a_config_rooted_in_the_profile_is_omission(tmp_path):
    """A remedy that cannot help is not a remedy (2026-08-09 blocker).

    When the conflicts come from a config whose ROOT already resolves to
    the named profile -- the wizard's own ladder, the LES campaign
    configs -- the old message offered two dead ends: "edit those keys
    to the profile values" (which would switch a PBL parameterization
    back ON over an LES nest's resolved turbulence) and "name the
    profile you meant -- --physics-profile P", where P was the very flag
    the caller had just passed.  The honest remedy on this path is the
    third one: omit the flag and the materializer publishes the config's
    own physics, every domain, unchanged.
    """

    config = _wizard_ladder_config(tmp_path)
    with pytest.raises(ValueError) as caught:
        runner._render_materialized_experiment(
            config.read_text(encoding="utf-8"), source="gfs",
            profile=runner.MORRISON_PHYSICS_PROFILE, origin=str(config))
    terse = str(caught.value).split("[[explain]]")[0]
    # Never the circle: the flag that was refused is not the remedy.
    assert "the shipped suite this config already IS" not in terse
    assert (f"name the profile you meant -- --physics-profile "
            f"{runner.MORRISON_PHYSICS_PROFILE}") not in terse
    # Never the flattening: these are the config's own deliberate
    # departures, not typos to be edited back to the profile.
    assert "edit those keys" not in terse
    # The remedy that works, and what taking it publishes.
    assert "omit --physics-profile" in terse
    rendered, exp, _receipt = runner._render_materialized_experiment(
        config.read_text(encoding="utf-8"), source="gfs", profile=None)
    nest = exp.domains[1].run
    # cu_physics is the departure that carries this now: since 2.5.0's
    # radt-floor repair the nest INHERITS the root's radt, so 12.0 here
    # is the config's own value published unchanged, not a flattening.
    assert (int(nest.cu_physics), float(nest.radt)) == (0, 12.0)


def test_a_declared_rrtmg_variant_is_governed_by_the_profile_resolution(
        tmp_path):
    """The reachable half of the unpinned-key judgement (2026-08-09).

    ``ra_rrtmg_variant`` selects which radiation IMPLEMENTATION a
    resolved (4, 4) pair executes -- gpuwm/core/rrtmg_legacy.py or
    gpuwm/core/rrtmgp.py.  Exactly one shipped profile pins the pair
    without pinning the variant (the 20CRv3 WSM6 suite, whose name
    asserts rte-rrtmgp); before this test, a config declaring
    ``ra_physics = 4`` plus ``ra_rrtmg_variant = "rrtmg_legacy"`` kept
    the variant under the kept-unpinned rule and published an authority
    running LEGACY RRTMG under that profile's name, silently, where the
    old delete-everything renderer produced rte-rrtmgp -- and step 5
    could never notice, because the key is outside the switch table.  A
    key the profile RESOLVES is governed like a key it pins.
    """

    silent = _asking_for_no_physics(
        (ROOT / "configs" / "gfs_wrf_direct_proof.toml").read_text(
            encoding="utf-8"))
    profile = runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE
    pinned = runner._profile_pinned_physics(
        runner._profile_runtime_switches("gfs", profile))
    # The hazard's precondition, pinned so a future profile edit that
    # closes it retires this test loudly instead of leaving it vacuous.
    assert "ra_rrtmg_variant" not in pinned
    assert runner._profile_radiation_pair(pinned) == (4, 4)

    legacy = _declaring_before_first_domain(
        silent, 'ra_physics = 4\nra_rrtmg_variant = "rrtmg_legacy"\n\n')
    with pytest.raises(
            ValueError,
            match=r"ra_rrtmg_variant: config 'rrtmg_legacy', "
                  r"profile 'rte-rrtmgp'"):
        runner._render_materialized_experiment(
            legacy, source="gfs", profile=profile)

    # Agreement with the resolved default stays the kept-unpinned rule:
    # the declaration survives where the user wrote it.
    agreeing = _declaring_before_first_domain(
        silent, 'ra_physics = 4\nra_rrtmg_variant = "rte-rrtmgp"\n\n')
    rendered, exp, _receipt = runner._render_materialized_experiment(
        agreeing, source="gfs", profile=profile)
    assert 'ra_rrtmg_variant = "rte-rrtmgp"' in rendered
    assert exp.root.run.ra_rrtmg_variant == "rte-rrtmgp"


def test_materializer_is_create_only_and_keeps_only_genuine_refusals(
        tmp_path):
    """Converted from the per-source profile whitelist (removed 2026-07-31).

    The owner ruled the suite choice is the user's: a registered shipped
    profile materializes for any prepared source now, so the old
    "not available for source" list refusal is gone.  What must remain
    fail-closed is the genuine blocker -- the registry's land-surface
    route declaration, which carries the GFS+RUC `mavail must be
    finite` field finding -- and the create-only output contract.
    """

    experiment = _no_physics_copy(
        tmp_path, ROOT / "configs" / "gfs_wrf_direct_proof.toml")
    wps = ROOT / "configs" / "gfs_wrf_direct_proof.namelist.wps"
    output = tmp_path / "authorities"
    runner.materialize_named_source_authorities(
        source="gfs", base_experiment_config=experiment,
        base_wps_namelist=wps, physics_profile=runner.MORRISON_PHYSICS_PROFILE,
        output_directory=output)
    with pytest.raises(FileExistsError):
        runner.materialize_named_source_authorities(
            source="gfs", base_experiment_config=experiment,
            base_wps_namelist=wps,
            physics_profile=runner.MORRISON_PHYSICS_PROFILE,
            output_directory=output)
    # A shipped profile that only ever ran on another source is admitted
    # -- reported as implemented-unverified, never refused by name.
    cross = runner.materialize_named_source_authorities(
        source="gfs", base_experiment_config=experiment,
        base_wps_namelist=wps,
        physics_profile=runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE,
        output_directory=tmp_path / "cross")
    assert cross["status"] == "PASS"
    assert cross["readiness"] == "IMPLEMENTED_UNVERIFIED"
    # The genuine per-source blocker refuses on the resolved selector,
    # names the registry declaration, and creates nothing.
    with pytest.raises(ValueError, match="does not offer the ruc-lsm"):
        runner.materialize_named_source_authorities(
            source="gfs", base_experiment_config=experiment,
            base_wps_namelist=wps,
            physics_profile=runner.RUC_PHYSICS_PROFILE,
            output_directory=tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()
    # A name the shipped tables genuinely do not define still fails
    # closed, naming the value.
    with pytest.raises(ValueError, match="not a shipped runner profile"):
        runner.materialize_named_source_authorities(
            source="gfs", base_experiment_config=experiment,
            base_wps_namelist=wps,
            physics_profile="no-such-suite-v1",
            output_directory=tmp_path / "unknown")
    assert not (tmp_path / "unknown").exists()


def test_materializer_cli_publishes_receipted_authorities(tmp_path, capsys):
    output = tmp_path / "cli-authorities"
    assert runner.main([
        "--materialize-authorities",
        "--source", "era5",
        "--base-experiment-config",
        str(_no_physics_copy(
            tmp_path, ROOT / "configs" / "era5_wrf_direct_proof.toml")),
        "--base-wps-namelist",
        str(ROOT / "configs" / "era5_wrf_direct_proof.namelist.wps"),
        "--physics-profile", runner.MORRISON_PHYSICS_PROFILE,
        "--output-directory", str(output),
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "PASS"
    assert summary["physics_profile"] == runner.MORRISON_PHYSICS_PROFILE
    assert Path(summary["receipt"]["path"]) \
        == (output / "authority-receipt.json").resolve()


def test_materializer_preserves_hierarchy_and_nondefault_explicit_vertical(
        tmp_path):
    base = tmp_path / "hierarchy.toml"
    source = _asking_for_no_physics(
        (ROOT / "configs" / "gfs_wrf_hierarchy_proof.toml").read_text(
            encoding="utf-8"))
    assert source.count("hybrid_opt = 2") == 1
    base.write_text(
        source.replace("hybrid_opt = 2", "hybrid_opt = 1"),
        encoding="utf-8")
    wps = ROOT / "configs" / "gfs_wrf_hierarchy_proof.namelist.wps"
    output = tmp_path / "hierarchy-authorities"

    receipt = runner.materialize_named_source_authorities(
        source="gfs", base_experiment_config=base,
        base_wps_namelist=wps,
        physics_profile=runner.MORRISON_PHYSICS_PROFILE,
        output_directory=output)
    generated = load_experiment(output / "experiment.toml")

    assert len(generated.domains) == 2
    assert generated.vertical.hybrid_opt == 1
    assert generated.vertical.eta_levels == load_experiment(base).vertical.eta_levels
    assert [domain.run.mp_physics for domain in generated.domains] == [10, 10]
    assert receipt["preserved"]["mass_levels"] == generated.root.run.nz
    assert receipt["preserved"]["hybrid_opt"] == 1


def test_materialized_descriptor_bytes_are_deterministic_across_case_roots(
        tmp_path):
    experiment = _no_physics_copy(
        tmp_path, ROOT / "configs" / "gfs_wrf_direct_proof.toml")
    wps = ROOT / "configs" / "gfs_wrf_direct_proof.namelist.wps"
    receipts = [
        runner.materialize_named_source_authorities(
            source="gfs", base_experiment_config=experiment,
            base_wps_namelist=wps,
            physics_profile=runner.MORRISON_PHYSICS_PROFILE,
            output_directory=tmp_path / f"case-{index}")
        for index in range(2)
    ]

    assert (tmp_path / "case-0" / "experiment.toml").read_bytes() \
        == (tmp_path / "case-1" / "experiment.toml").read_bytes()
    assert receipts[0]["generated"]["experiment_config"]["sha256"] \
        == receipts[1]["generated"]["experiment_config"]["sha256"]
    assert receipts[0]["normalized_selected_physics"] \
        == receipts[1]["normalized_selected_physics"]


def _prepared_fixture(
        tmp_path: Path, source: str, *, adapter=None, hierarchy=False,
        physics_profile=runner.PHYSICS_PROFILE,
        twentycr_decoder_roles=frozenset({"gpuwm_mapped_engine"}),
):
    if hierarchy and source not in {"gfs", "20crv3"}:
        raise ValueError("synthetic hierarchy fixture uses GFS-shaped configs")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    kind = "hierarchy" if hierarchy else "direct"
    config_source = "gfs" if source == "20crv3" else source
    config_name = f"{config_source}_wrf_{kind}_proof.toml"
    wps_name = f"{config_source}_wrf_{kind}_proof.namelist.wps"
    experiment_config = tmp_path / config_name
    wps_namelist = tmp_path / wps_name
    shutil.copy2(ROOT / "configs" / config_name, experiment_config)
    shutil.copy2(ROOT / "configs" / wps_name, wps_namelist)
    config_text = experiment_config.read_text(encoding="utf-8")
    if physics_profile is not None:
        # The base asks for no physics, so the named profile SUPPLIES the
        # suite instead of overwriting a declared one; see
        # _asking_for_no_physics.  Byte-identical result, new contract.
        config_text = _asking_for_no_physics(config_text)
    if physics_profile == runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE:
        if source != "20crv3":
            raise ValueError("synthetic 20CRv3 profile requires source=20crv3")
        assert config_text.count("history_interval_s = 3600.0\n") == 1
        config_text = config_text.replace(
            "history_interval_s = 3600.0\n",
            "history_interval_s = 10800.0\n",
        )
    rendered, _exp, _receipt = runner._render_materialized_experiment(
        config_text, source=source, profile=physics_profile)
    experiment_config.write_text(rendered, encoding="utf-8")
    exp = load_experiment(experiment_config)
    forcing_hours = [0, 3] if source in {"gfs", "20crv3"} else [0, 6, 12]
    preprocessing = {
        "schema": "gpuwm-preprocess-implementation-v2",
        "backend": "cpu",
        "implementation": "test-source-neutral-preprocess",
        "workers": 2,
    }
    bridge_digest = hashlib.sha256(f"{source}-bridge".encode()).hexdigest()
    files = {}
    if source != "20crv3":
        files.update({
            "bridge": {
                "name": f"{source}_bridge.exe", "sha256": bridge_digest},
            "experiment_config": {
                "name": experiment_config.name,
                "sha256": _sha256(experiment_config),
            },
            "wps_namelist": {
                "name": wps_namelist.name,
                "sha256": _sha256(wps_namelist),
            },
        })
    if source == "gfs":
        files["series"] = {
            "name": "gfs-series.tsv",
            "sha256": hashlib.sha256(b"series").hexdigest(),
        }
        for hour in forcing_hours:
            files[f"grib-f{hour:03d}"] = {
                "name": f"gfs.f{hour:03d}.grib2",
                "sha256": hashlib.sha256(f"gfs-{hour}".encode()).hexdigest(),
            }
        manifest = {
            "schema": runner._SOURCE_SCHEMA[source],
            "source": {
                "model": "GFS",
                "product": "pgrb2.0p25",
                "cycle": exp.start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "files": files,
        }
    elif source == "era5":
        files.update({
            "grib": {
                "name": "era5.grib1",
                "sha256": hashlib.sha256(b"era5-grib").hexdigest(),
            },
            "vtable": {
                "name": "Vtable.ERA5",
                "sha256": hashlib.sha256(b"era5-vtable").hexdigest(),
            },
        })
        manifest = {
            "schema": runner._SOURCE_SCHEMA[source],
            "files": files,
        }
    else:
        member = "072"
        rows = []
        for hour in forcing_hours:
            valid_time = exp.start_time + timedelta(hours=hour)
            for role in ("pl", "sfc"):
                filename = (
                    f"mem{member}_{valid_time:%Y%m%d%H}_{role}.grb2")
                rows.append({
                    "path": str((tmp_path / "raw" / filename).resolve()),
                    "filename": filename,
                    "member": member,
                    "valid_time": valid_time.isoformat(),
                    "role": role,
                    "bytes": 1000 + hour + (1 if role == "sfc" else 0),
                    "sha256": hashlib.sha256(filename.encode()).hexdigest(),
                })
        manifest = {
            "schema": runner._SOURCE_SCHEMA[source],
            "source": runner._TWENTYCRV3_SOURCE,
            "member_identity": runner._TWENTYCRV3_MEMBER_IDENTITY,
            "member": member,
            "valid_times": [
                (exp.start_time + timedelta(hours=hour)).isoformat()
                for hour in forcing_hours],
            "cadence_seconds": 10800,
            "file_count": len(rows),
            "files": rows,
        }
        manifest["content_sha256"] = hashlib.sha256(
            _canonical(manifest).encode()).hexdigest()
    if source == "20crv3":
        evidence = prepared / "source-evidence"
        evidence.mkdir()
        authorities = runner.twentycrv3_authority_sha256()
        authority_root = ROOT / "gpuwm" / "authorities"
        shutil.copy2(
            authority_root / "rw-wps-20crv3-member-grib2.mapping.json",
            evidence / "mapping.json")
        shutil.copy2(
            authority_root / "rw-wps-20crv3-member-grib2.composition.json",
            evidence / "composition.json")
        shutil.copy2(
            authority_root / "rw-wps-20crv3-member-grib2.provenance.json",
            evidence / "provenance-test.json")
        assert _sha256(evidence / "mapping.json") == authorities["mapping"]
        assert _sha256(evidence / "composition.json") \
            == authorities["composition"]
        assert _sha256(evidence / "provenance-test.json") \
            == authorities["provenance"]
        source_manifest = evidence / "input-manifest.json"
    else:
        source_manifest = prepared / "source-input-manifest.json"
    _write_json(source_manifest, manifest)
    manifest_digest = _sha256(source_manifest)
    bridged_digest = None
    if source == "20crv3":
        # The bridged composition-inputs document the post-port writer
        # publishes beside the member manifest, so the reader can
        # re-hash the receipt's input_manifest record against bytes.
        bridged = prepared / "source-evidence" / "composition-inputs.json"
        _write_json(bridged, {
            "schema": "gpuwm-mapped-composition-inputs-v1",
            "fixture": "bridged-composition-inputs",
            "member": manifest["member"],
            "member_identity": manifest["member_identity"],
        })
        bridged_digest = _sha256(bridged)
    mapped_authorities = None
    source_composition = None
    target_contract = None
    decoder_digests = None
    if source == "20crv3":
        mapped_authorities = dict(runner.twentycrv3_authority_sha256())
        decoder_digests = {
            role: hashlib.sha256(f"20crv3-{role}".encode()).hexdigest()
            for role in twentycr_decoder_roles
        }
        decoder_paths = {
            role: str((tmp_path / f"{role}.exe").resolve())
            for role in twentycr_decoder_roles
        }
        composition_document = json.loads(
            (prepared / "source-evidence" / "composition.json").read_text(
                encoding="utf-8"))
        source_composition = {
            "schema": "gpuwm-mapped-composition-receipt-v1",
            "status": "CANONICAL_FRAMES_COMPLETE_NOT_STOCK_WRF_CERTIFIED",
            "mapping": {
                "path": str((tmp_path / "mapping.json").resolve()),
                "sha256": mapped_authorities["mapping"],
            },
            "composition": {
                "path": str((tmp_path / "composition.json").resolve()),
                "sha256": mapped_authorities["composition"],
            },
            "input_manifest": {
                "path": str((tmp_path / "composition-inputs.json").resolve()),
                "sha256": bridged_digest,
            },
            "decoders": {
                role: {"path": decoder_paths[role],
                       "sha256": decoder_digests[role]}
                for role in sorted(twentycr_decoder_roles)
            },
            "terrain_products": [
                {"path": row["path"], "sha256": row["sha256"]}
                for row in manifest["files"] if row["role"] == "sfc"
            ],
            "terrain_provenance": {
                "provenance_path": str((tmp_path / "provenance.json").resolve()),
                "provenance_sha256": mapped_authorities["provenance"],
            },
            # The composed exact-subset receipt the post-port writer
            # seals: mapped_composition's terrain receipt plus the
            # ensemble identity the member manifest bound.
            "alignment": {
                "schema": "gpuwm-mapped-exact-subset-binding-v1",
                "status": "PASS",
                "field": "terrain_height",
                "primary_shape": [181, 360],
                "supplement_shape": [181, 360],
                "latitude_index_range": [0, 180],
                "longitude_index_range": [0, 359],
                "latitude_sha256": hashlib.sha256(
                    b"20crv3-latitude").hexdigest(),
                "longitude_sha256": hashlib.sha256(
                    b"20crv3-longitude").hexdigest(),
                "terrain_full_sha256": hashlib.sha256(
                    b"20crv3-terrain-full").hexdigest(),
                "terrain_subset_sha256": hashlib.sha256(
                    b"20crv3-terrain-subset").hexdigest(),
                "supplement_valid_times": manifest["valid_times"],
                "matched_primary_valid_times": manifest["valid_times"],
                "invariant_across_all_supplement_times": True,
                "latitude_index_direction": 1,
                "longitude_index_direction": 1,
                "coordinate_match": "exact_equivalent_contiguous_subset",
                "longitude_equivalence": "modulo_360_exact",
                "member": manifest["member"],
                "member_identity": manifest["member_identity"],
            },
            "soil_layers": composition_document["soil_layers"],
            "frame_count": len(forcing_hours),
            "valid_times": manifest["valid_times"],
            "frames": [{
                "header_sha256": hashlib.sha256(
                    f"header-{hour}".encode()).hexdigest(),
                "terrain_sha256": hashlib.sha256(
                    f"terrain-{hour}".encode()).hexdigest(),
                "field_count": 15,
            } for hour in forcing_hours],
        }
        source_composition["receipt_content_sha256"] = hashlib.sha256(
            _canonical(source_composition).encode()).hexdigest()
        target_contract = {
            "schema": "gpuwm-mapped-target-contract-v1",
            "max_dom": len(exp.domains),
            "boundary_interval_seconds": 10800,
        }

    domain_bundle = prepared
    if hierarchy:
        domain_bundle = prepared / "hierarchy-artifacts" / "domains" / "d01"
        domain_bundle.mkdir(parents=True)
    static_path = domain_bundle / "native-static.npz"
    if source == "20crv3":
        np.savez(static_path, STATIC=np.ones((1,), dtype=np.float64))
    else:
        static_path.write_bytes(b"hash-bound-test-static")
    geometry_path = domain_bundle / "geometry-receipt.json"
    geometry_payload = {
        "schema": "gpuwm-native-static-direct-v1",
        "status": "PASS",
        "geometry": {"test": source},
        "cache": {
            "path": static_path.name,
            "bytes": static_path.stat().st_size,
            "sha256": _sha256(static_path),
        },
    }
    _write_json(geometry_path, geometry_payload)

    if source == "20crv3":
        source_identity = {
            "adapter": adapter or runner._SOURCE_ADAPTER[source],
            "mapping_sha256": mapped_authorities["mapping"],
            "composition_sha256": mapped_authorities["composition"],
            "input_manifest_sha256": manifest_digest,
            "composition_receipt_sha256": source_composition[
                "receipt_content_sha256"],
            "preprocessing": preprocessing,
        }
        if hierarchy:
            source_identity.update({
                "target_contract": target_contract,
                "nested_source_orography": {"schema": "test-orography-v1"},
                "hierarchy_implementation_sha256": {
                    "test": hashlib.sha256(b"hierarchy").hexdigest()},
            })
    else:
        source_identity = {
            "adapter": adapter or runner._SOURCE_ADAPTER[source],
            "input_manifest_schema": runner._SOURCE_SCHEMA[source],
            "input_manifest_sha256": manifest_digest,
            "decoder": {
                "name": files["bridge"]["name"],
                "sha256": bridge_digest,
                "implementation": runner._DECODER_IMPLEMENTATION[source],
            },
            "preprocessing": preprocessing,
        }
    if source == "gfs":
        source_identity.update({
            "implementation_sha256": {"test": "gfs"},
            "git_source_identity": {"commit": "test-gfs"},
        })
    if hierarchy:
        source_identity["grid_id"] = 1
    identity = prepared_cache_identity(
        bridge_manifest_sha256=manifest_digest,
        source_manifest_sha256=manifest_digest,
        static_cache_sha256=_sha256(static_path),
        namelist_sha256=_sha256(experiment_config),
        domain_config=exp.root,
        forcing_hours=forcing_hours,
        source_identity=source_identity,
    )
    cache = domain_bundle / "prepared-cache"
    cache.mkdir()
    array = np.asarray([1.0], dtype=np.float32)
    with (cache / "a00000.npy").open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    arrays = {
        "test": {
            "file": "a00000.npy",
            "shape": [1],
            "dtype": "float32",
            "nbytes": 4,
            "sha256": _array_sha256(array),
        },
    }
    intervals = [
        {
            "start_seconds": float(first * 3600),
            "end_seconds": float(second * 3600),
            "fields": list(runner._LBC_FIELDS),
        }
        for first, second in zip(forcing_hours, forcing_hours[1:])
    ]
    user_metadata = {
            "initial_valid_time": exp.start_time.isoformat(),
            "last_valid_time": (
                exp.start_time + timedelta(hours=forcing_hours[-1])).isoformat(),
            "forcing_hours": forcing_hours,
    }
    if not hierarchy:
        if source == "20crv3":
            user_metadata.update({
                "source_adapter": "mapped",
                "boundary_interval_seconds": (
                    forcing_hours[1] - forcing_hours[0]) * 3600,
                "composition_receipt_sha256": source_composition[
                    "receipt_content_sha256"],
            })
        else:
            user_metadata.update({
                "source_adapter": source,
                "boundary_interval_seconds": (
                    forcing_hours[1] - forcing_hours[0]) * 3600,
                "preprocessing": preprocessing,
            })
    elif source == "20crv3":
        user_metadata.update({
            "composition_receipt_sha256": source_composition[
                "receipt_content_sha256"],
            "mapped_target_contract": target_contract,
        })
    metadata = {
        "user": user_metadata,
        "state_names": [],
        "coord_arrays": [],
        "coord_scalars": {},
        "base_arrays": [],
        "base_scalars": {},
        "met_fields": sorted(runner._REQUIRED_MET_FIELDS),
        "surface_fields": sorted(runner._CANONICAL_SURFACE_FIELDS),
        "lbc": {
            "spec_bdy_width": exp.root.run.spec_bdy_width,
            "spec_zone": exp.root.run.spec_zone,
            "relax_zone": exp.root.run.relax_zone,
            "intervals": intervals,
        },
        "setup_fingerprint": "test",
    }
    basis = {
        "schema": PREPARED_CACHE_SCHEMA,
        "identity": identity,
        "metadata": metadata,
        "arrays": arrays,
        "payload_bytes": 4,
    }
    header = {
        **basis,
        "status": "READY",
        "created_utc": "2026-07-23T00:00:00+00:00",
        "content_sha256": hashlib.sha256(
            _canonical(basis).encode()).hexdigest(),
    }
    header_path = cache / "header.json"
    _write_json(header_path, header)

    boundary_seconds = (forcing_hours[1] - forcing_hours[0]) * 3600
    export_source = {
        "contract_sha256": _sha256(
            ROOT / "gpuwm" / "wrf_direct_v461_contract.json"),
        "geometry_receipt_sha256": _sha256(geometry_path),
        "prepared_content_sha256": header["content_sha256"],
        "prepared_header_sha256": _sha256(header_path),
        "resolved_physics_contract_sha256": (
            runner._resolved_wrf_direct_contract_sha256(
                exp.root.run.mp_physics)),
        "static_cache_sha256": _sha256(static_path),
    }
    cache_receipt = {
        "schema": PREPARED_CACHE_SCHEMA,
        "status": "BUILT",
        "path": "prepared-cache",
        "content_sha256": header["content_sha256"],
        "array_count": 1,
        "payload_bytes": 4,
    }
    front_door_physics = None
    if source == "gfs" and not hierarchy:
        acknowledgements = (
            ("noahmp-host-column-throughput-v1",)
            if physics_profile == runner.NOAHMP_PHYSICS_PROFILE else ()
        )
        if physics_profile is None:
            # Cross-lane: the proof's physics receipt comes from the
            # REAL front door (the other lane's writer), not from a
            # fixture re-spelling of it.
            from gpuwm.gfs_direct import front_door_physics_selection

            front_door_physics = front_door_physics_selection(exp)
        else:
            front_door_physics = (
                runner.validate_single_domain_physics_profile(
                    physics_profile, config=exp.root.run,
                    expert_acknowledgements=acknowledgements))
    proof = {
        "schema": runner._PROOF_SCHEMA[source],
        "status": "READY_NOT_YET_STOCK_WRF_GATED",
        "forcing_times": [
            (exp.start_time + timedelta(hours=hour)).isoformat()
            for hour in forcing_hours
        ],
        "forcing_hours": forcing_hours,
        "boundary_interval_seconds": boundary_seconds,
        "input_manifest_sha256": manifest_digest,
        "decoder_sha256": bridge_digest,
        "preprocessing": preprocessing,
        "preprocessing_receipt_sha256": hashlib.sha256(
            _canonical(preprocessing).encode()).hexdigest(),
        "source_inputs": {
            "manifest_schema": manifest["schema"],
            "manifest_sha256": manifest_digest,
            "files": manifest["files"],
        },
        "initialization_artifacts": {
            "source_manifest": _artifact(
                source_manifest, "source-input-manifest.json"),
            "static_cache": _artifact(static_path, "native-static.npz"),
            "geometry_receipt": _artifact(
                geometry_path, "geometry-receipt.json"),
            "prepared_cache": {
                "path": "prepared-cache",
                "content_sha256": header["content_sha256"],
                "payload_bytes": 4,
            },
            "wrf_files": {},
        },
        "prepared_cache": cache_receipt,
        "export": {
            "schema": (
                "gpuwm-native-direct-wrf-export-v3"
                if source == "gfs"
                else "gpuwm-native-direct-wrf-export-v2"),
            "status": "READY",
            "forcing_hours": forcing_hours,
            "boundary_interval_seconds": boundary_seconds,
            "dimensions": {
                "nx": exp.root.run.nx,
                "ny": exp.root.run.ny,
                "nz": exp.root.run.nz,
            },
            "valid_time": exp.start_time.strftime("%Y-%m-%d_%H:%M:%S"),
            "source": export_source,
        },
    }
    if front_door_physics is not None:
        proof["physics"] = front_door_physics
        proof["export"]["physics"] = front_door_physics
    if source == "20crv3":
        proof = {
            "schema": runner._PROOF_SCHEMA[source],
            "status": "READY_NOT_YET_STOCK_WRF_GATED",
            "forcing_times": manifest["valid_times"],
            # WRITTEN BY THE REAL WRITER, so it belongs in the fixture.
            # gpuwm/mapped_direct.py puts this key in every mapped proof
            # it publishes -- direct and hierarchy alike -- and this
            # fixture omitting it is what let the runner's exact
            # top-level key check drift out of agreement with the
            # preparation that feeds it.  A reader tested against its
            # own fixture instead of the other lane's real writer is
            # exactly the trap this line closes.
            "soil_texture_downscale": {},
            "forcing_hours": forcing_hours,
            "boundary_interval_seconds": boundary_seconds,
            "execution_inputs": {
                "decoders": {
                    role: {
                        "path": source_composition["decoders"][role]["path"],
                        "bytes": 1024 + index,
                        "sha256": decoder_digests[role],
                    }
                    for index, role in enumerate(
                        sorted(twentycr_decoder_roles))
                },
                "experiment_config": _artifact(
                    experiment_config, str(experiment_config.resolve())),
                "wps_namelist": _artifact(
                    wps_namelist, str(wps_namelist.resolve())),
                "geog_root": str((tmp_path / "geog").resolve()),
                "geog_source_binding": (
                    "resolved_dataset_paths_plus_native_static_output_sha256"),
                "geog_resolution_tokens": ["default"],
                "geog_datasets": {},
            },
            "source_composition": source_composition,
            "preprocessing": preprocessing,
            "static": {
                "path": "native-static.npz",
                "bytes": static_path.stat().st_size,
                "sha256": _sha256(static_path),
                "fields": ["STATIC"],
            },
            "geometry": geometry_payload,
            "prepared_cache": cache_receipt,
            "export": {
                "schema": "gpuwm-native-direct-wrf-export-v2",
                "status": "READY",
                "forcing_hours": forcing_hours,
                "boundary_interval_seconds": boundary_seconds,
                "dimensions": {
                    "nx": exp.root.run.nx,
                    "ny": exp.root.run.ny,
                    "nz": exp.root.run.nz,
                },
                "valid_time": exp.start_time.strftime("%Y-%m-%d_%H:%M:%S"),
                "source": export_source,
            },
            "timing_seconds": {"total": 1.0},
        }
    if hierarchy:
        domain_receipt = {
            "schema": "gpuwm-native-domain-artifact-build-v1",
            "status": "READY",
            "grid_id": 1,
            "parent_id": 0,
            "boundary_mode": "external-specified",
            "valid_time": exp.start_time.isoformat(),
            "forcing_hours": forcing_hours,
            "artifacts": {
                "prepared_cache": {
                    "path": "prepared-cache",
                    "content_sha256": header["content_sha256"],
                    "payload_bytes": 4,
                    "array_count": 1,
                },
                "static_cache": {
                    "path": "native-static.npz",
                    "bytes": static_path.stat().st_size,
                    "sha256": _sha256(static_path),
                    "fields": ["STATIC"],
                },
                "geometry_receipt": {
                    "path": "geometry-receipt.json",
                    "sha256": _sha256(geometry_path),
                    "geometry": geometry_payload["geometry"],
                },
            },
            "verification": {
                "schema": PREPARED_CACHE_SCHEMA,
                "status": "PASS",
                "path": "prepared-cache",
                "content_sha256": header["content_sha256"],
                "array_count": 1,
                "payload_bytes": 4,
            },
        }
        _write_json(domain_bundle / "receipt.json", domain_receipt)
        hierarchy_root = prepared / "hierarchy-artifacts"
        artifact_manifest = {
            "schema": "gpuwm-native-domain-artifacts-v1",
            "domains": [{
                "grid_id": domain.grid_id,
                "prepared_cache": (
                    f"domains/d{domain.grid_id:02d}/prepared-cache"),
                "static_cache": (
                    f"domains/d{domain.grid_id:02d}/native-static.npz"),
                "geometry_receipt": (
                    f"domains/d{domain.grid_id:02d}/geometry-receipt.json"),
            } for domain in exp.domains],
        }
        artifact_manifest_path = hierarchy_root / "domain-artifacts.json"
        _write_json(artifact_manifest_path, artifact_manifest)
        hierarchy_receipt = {
            "schema": "gpuwm-native-hierarchy-artifact-build-v1",
            "status": "READY",
            "domain_count": len(exp.domains),
            "grid_ids": [domain.grid_id for domain in exp.domains],
            "manifest": {
                "path": "domain-artifacts.json",
                "sha256": _sha256(artifact_manifest_path),
            },
            "boundary_inventory": {
                "external": [1],
                "nested_parent_forced": [
                    domain.grid_id for domain in exp.domains[1:]],
            },
            "domains": [domain_receipt] + [
                {"grid_id": domain.grid_id} for domain in exp.domains[1:]],
        }
        _write_json(hierarchy_root / "receipt.json", hierarchy_receipt)
        d01_source = {
            key: value for key, value in export_source.items()
            if key != "contract_sha256"
        }
        d01_source.update({
            "mp_physics": exp.root.run.mp_physics,
            "microphysics": runner.stock_wrf_physics_inventory(
                exp.root.run.mp_physics).scheme,
        })
        wrf_manifest = {
            "schema": "gpuwm-native-direct-wrf-hierarchy-export-v1",
            "status": "READY",
            "valid_time": exp.start_time.strftime("%Y-%m-%d_%H:%M:%S"),
            "forcing_hours": forcing_hours,
            "boundary_interval_seconds": boundary_seconds,
            "hierarchy": runner._hierarchy_rows(exp),
            "source": {
                "contract_sha256": export_source["contract_sha256"],
                "domains": {
                    **{"d01": d01_source},
                    **{
                        f"d{domain.grid_id:02d}": {}
                        for domain in exp.domains[1:]
                    },
                },
                "input_provenance": {
                    "input_manifest_sha256": manifest_digest,
                    "decoder_sha256": bridge_digest,
                    "preprocessing": preprocessing,
                    "regular_source_adapter": source,
                    "native_artifact_manifest": (
                        "../hierarchy-artifacts/domain-artifacts.json"),
                    "native_artifact_manifest_sha256": _sha256(
                        artifact_manifest_path),
                },
            },
        }
        if source == "20crv3":
            wrf_manifest["source"]["input_provenance"] = {
                "mapping_sha256": mapped_authorities["mapping"],
                "composition_sha256": mapped_authorities["composition"],
                "input_manifest_sha256": manifest_digest,
                "decoder_sha256": decoder_digests,
                "preprocessing": preprocessing,
                "mapped_target_contract": target_contract,
                "regular_source_adapter": "rw-wps-mapped",
                "regular_source_forcing_times": manifest["valid_times"],
                "regular_source_static_catalog": {"schema": "test-catalog-v1"},
                "regular_source_orography": {"schema": "test-orography-v1"},
                "regular_source_coverage": {"schema": "test-coverage-v1"},
                "regular_source_topology": {"schema": "test-topology-v1"},
                "regular_source_hierarchy_implementation_sha256": {
                    "test": hashlib.sha256(b"hierarchy").hexdigest()},
                "native_artifact_manifest": (
                    "../hierarchy-artifacts/domain-artifacts.json"),
                "native_artifact_manifest_sha256": _sha256(
                    artifact_manifest_path),
            }
            shutil.copy2(static_path, prepared / "native-static.npz")
            shutil.copy2(geometry_path, prepared / "geometry-receipt.json")
        wrf_root = prepared / "wrf-native-input"
        wrf_root.mkdir()
        _write_json(wrf_root / "manifest.json", wrf_manifest)
        proof = {
            "schema": runner._HIERARCHY_PROOF_SCHEMA[source],
            "status": "READY_NOT_YET_STOCK_WRF_GATED",
            "domain_count": len(exp.domains),
            "forcing_times": [
                (exp.start_time + timedelta(hours=hour)).isoformat()
                for hour in forcing_hours
            ],
            "forcing_hours": forcing_hours,
            "boundary_interval_seconds": boundary_seconds,
            "input_manifest_sha256": manifest_digest,
            "decoder_sha256": bridge_digest,
            "preprocessing": preprocessing,
            "artifact_receipt": hierarchy_receipt,
            "wrf_manifest": wrf_manifest,
        }
        if source == "20crv3":
            proof = {
                "schema": runner._HIERARCHY_PROOF_SCHEMA[source],
                "status": "READY_NOT_YET_STOCK_WRF_GATED",
                "domain_count": len(exp.domains),
                "forcing_times": manifest["valid_times"],
                # As above: the real writer puts this in the hierarchy
                # proof too, so the fixture must.
                "soil_texture_downscale": {},
                "forcing_hours": forcing_hours,
                "boundary_interval_seconds": boundary_seconds,
                "target_contract": target_contract,
                "execution_inputs": {
                    "decoders": {
                        role: {
                            "path": source_composition["decoders"][role]["path"],
                            "bytes": 1024 + index,
                            "sha256": decoder_digests[role],
                        }
                        for index, role in enumerate(
                            sorted(twentycr_decoder_roles))
                    },
                    "experiment_config": _artifact(
                        experiment_config, str(experiment_config.resolve())),
                    "wps_namelist": _artifact(
                        wps_namelist, str(wps_namelist.resolve())),
                    "geog_root": str((tmp_path / "geog").resolve()),
                    "geog_source_binding": (
                        "per_domain_resolved_dataset_paths_plus_native_"
                        "static_output_sha256"),
                },
                "source_composition": source_composition,
                "preprocessing": preprocessing,
                "hierarchy_workers": 2,
                "root_static": {
                    "path": "native-static.npz",
                    "bytes": (prepared / "native-static.npz").stat().st_size,
                    "sha256": _sha256(prepared / "native-static.npz"),
                    "fields": ["STATIC"],
                },
                "root_geometry": geometry_payload,
                "static_catalog": {"schema": "test-catalog-v1"},
                "source_coverage": {"schema": "test-coverage-v1"},
                "artifact_receipt": hierarchy_receipt,
                "wrf_manifest": wrf_manifest,
                "timing_seconds": {"total": 1.0},
            }
    if source == "gfs":
        proof.update({
            "implementation_sha256": source_identity["implementation_sha256"],
            "git_source_identity": source_identity["git_source_identity"],
        })
    if source == "20crv3":
        proof["proof_content_sha256"] = hashlib.sha256(
            _canonical(proof).encode()).hexdigest()
    proof_path = prepared / "proof.json"
    _write_json(proof_path, proof)
    return SimpleNamespace(
        source=source, prepared=prepared, proof=proof_path,
        domain_bundle=domain_bundle,
        source_manifest=source_manifest, experiment=experiment_config,
        wps=wps_namelist, content_sha256=header["content_sha256"],
        run_seconds=exp.run_seconds)


@pytest.mark.parametrize("source", ("gfs", "era5", "20crv3"))
def test_gpuwm_sim_composes_the_command_a_real_bundle_is_accepted_with(
        tmp_path, monkeypatch, source):
    """THE end-to-end proof of the simulation seam, short of a GPU.

    ``gpuwm sim`` claims it can take a prepared tree, work out for
    itself which source produced it and which runner arm it needs, relay
    the digests off the bundle's own artifacts, and hand the runner
    something that runs.  Every part of that claim is checkable here
    without a card: build a bundle this runner's own preflight ACCEPTS,
    ask the seam to compose its command, then feed the values PARSED OUT
    OF THAT COMMAND back into the same preflight.

    A pass means the seam's published boundary -- what
    ``--print-command`` prints, and what a third party's script would
    copy -- drives a bundle the runner genuinely accepts.  A
    seam that composed a plausible-looking command nobody had run
    through the real gate would be the "engine-proven is not shipped"
    failure wearing a test.
    """

    fixture = _prepared_fixture(tmp_path, source)
    grid = SimpleNamespace(source=source)
    monkeypatch.setattr(
        runner, "validate_native_lambert_contract",
        lambda exp, path, *, source_name: grid)
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {
            "STATIC": np.ones((ny, nx), dtype=np.float64)})

    from gpuwm import stage_cli

    bundle = stage_cli.resolve_bundle(fixture.prepared)
    assert bundle["source"] == source, (
        "the seam read the wrong source off the bundle's own document")
    assert bundle["layout"] == "single"

    profile = runner.PHYSICS_PROFILE
    command = stage_cli.sim_command(
        bundle, experiment_config=fixture.experiment,
        wps_namelist=fixture.wps, outdir=tmp_path / "run",
        physics_profile=profile)
    flags = {command[i]: command[i + 1]
             for i in range(3, len(command) - 1)
             if command[i].startswith("--")}

    # The relay read the digests off the artifacts.  Independently
    # recomputing them here is the cross-check: `sim` takes the two
    # inside the proof, and the fixture hashes the files, and a valid
    # bundle is exactly one where those agree.
    assert flags["--proof-sha256"] == _sha256(fixture.proof)
    assert flags["--source-manifest-sha256"] \
        == _sha256(fixture.source_manifest)
    assert flags["--prepared-content-sha256"] == fixture.content_sha256

    inputs = runner.preflight_prepared_forecast(
        source=flags["--source"],
        prepared_root=Path(flags["--prepared-root"]),
        proof_sha256=flags["--proof-sha256"],
        source_manifest_sha256=flags["--source-manifest-sha256"],
        prepared_content_sha256=flags["--prepared-content-sha256"],
        experiment_config=Path(flags["--experiment-config"]),
        wps_namelist=Path(flags["--wps-namelist"]),
        physics_profile=flags["--physics-profile"],
        run_seconds=fixture.run_seconds,
        history_interval_seconds=load_experiment(
            fixture.experiment).root.history_interval_s)

    assert inputs.source == source
    assert inputs.cache_reader.verify_all()["status"] == "PASS"
    runner._verify_inputs_unchanged(inputs)


@pytest.mark.parametrize("source", ("gfs", "era5"))
def test_preflight_accepts_exact_portable_single_domain_authorities(
        tmp_path, monkeypatch, source):
    fixture = _prepared_fixture(tmp_path, source)
    grid = SimpleNamespace(source=source)
    monkeypatch.setattr(
        runner, "validate_native_lambert_contract",
        lambda exp, path, *, source_name: grid)
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {"STATIC": np.ones((ny, nx))})

    inputs = runner.preflight_prepared_forecast(
        source=source, prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds, history_interval_seconds=3600)

    assert inputs.source == source
    assert inputs.cache_reader.verify_all()["status"] == "PASS"
    assert inputs.physics_receipt["profile"] == runner.PHYSICS_PROFILE
    assert dict(inputs.landuse_identity) == dict(runner._LANDUSE_IDENTITY)
    assert inputs.forcing_hours == ((0, 3) if source == "gfs" else (0, 6, 12))


def _bind_synthetic_preflight_geometry(monkeypatch, *, hierarchy: bool):
    grid = SimpleNamespace(source="20crv3")
    if hierarchy:
        monkeypatch.setattr(
            runner, "validate_native_lambert_contracts",
            lambda exp, path, *, source_name: tuple(
                grid for _ in exp.domains))
    else:
        monkeypatch.setattr(
            runner, "validate_native_lambert_contract",
            lambda exp, path, *, source_name: grid)
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {
            "STATIC": np.ones((ny, nx), dtype=np.float64)})


def _preflight_fixture(fixture, *, physics_profile=runner.PHYSICS_PROFILE,
                       expert_acknowledgements=()):
    history_interval_seconds = load_experiment(
        fixture.experiment).root.history_interval_s
    return runner.preflight_prepared_forecast(
        source=fixture.source, prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=physics_profile,
        expert_acknowledgements=tuple(expert_acknowledgements),
        run_seconds=fixture.run_seconds,
        history_interval_seconds=history_interval_seconds)


def _preflight_fixture_with_proof_sha(fixture, proof_sha256: str):
    """``_preflight_fixture`` with one digest replaced, nothing else."""

    history_interval_seconds = load_experiment(
        fixture.experiment).root.history_interval_s
    return runner.preflight_prepared_forecast(
        source=fixture.source, prepared_root=fixture.prepared,
        proof_sha256=proof_sha256,
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds,
        history_interval_seconds=history_interval_seconds)


def test_preflight_accepts_exact_member_20crv3_mapped_direct_d01(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "20crv3")
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)

    inputs = _preflight_fixture(fixture)

    assert inputs.layout == "mapped-direct-d01-v1"
    assert inputs.source_member == "072"
    assert inputs.forcing_hours == (0, 3)
    assert inputs.boundary_interval_seconds == 10800
    assert inputs.cache_identity["source_identity"]["adapter"] \
        == "rw-wps-20crv3-member-grib2-v1"
    assert {"mapped_mapping", "mapped_composition", "mapped_provenance"} \
        <= set(inputs.authority_paths)
    runner._verify_inputs_unchanged(inputs)


def test_20crv3_accepts_the_python_engine_decoder_pair_in_full(
        tmp_path, monkeypatch):
    """The workaround shape: a preparation sealed on the subprocess pair.

    The member manifest declares no decoder section, so the reader pins
    _TWENTYCRV3_DECODER_ROLE_SETS: the in-process engine OR the
    subprocess pair, each in full.  This is the pair half; a partial
    inventory (a role from each) is refused below rather than assumed.
    """

    fixture = _prepared_fixture(
        tmp_path, "20crv3",
        twentycr_decoder_roles=frozenset(
            {"grib2_inventory", "grib2_dump"}))
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)

    inputs = _preflight_fixture(fixture)

    assert inputs.source_member == "072"
    runner._verify_inputs_unchanged(inputs)


def test_20crv3_refuses_a_decoder_inventory_that_is_neither_route(
        tmp_path, monkeypatch):
    """One role from each route is a run nothing shipped: refused by name."""

    fixture = _prepared_fixture(
        tmp_path, "20crv3",
        twentycr_decoder_roles=frozenset(
            {"gpuwm_mapped_engine", "grib2_dump"}))
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)

    with pytest.raises(
            ValueError,
            match="neither the in-process engine nor the subprocess pair"):
        _preflight_fixture(fixture)


def test_20crv3_implemented_unverified_profile_retains_hash_bound_cadence(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(
        tmp_path, "20crv3",
        physics_profile=runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)

    inputs = _preflight_fixture(
        fixture, physics_profile=runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE)

    assert inputs.physics_receipt["readiness"] == "IMPLEMENTED_UNVERIFIED"
    assert inputs.physics_receipt["resolved"]["radiation_scheme_ids"] == [4, 4]
    plan = inputs.physics_receipt["execution_plan"]
    assert plan["physics_overrides"] == []
    assert plan["source_experiment"]["domains"][0][
        "history_interval_seconds"] == 10800.0
    assert plan["executed_d01"]["domain"][
        "history_interval_seconds"] == 10800.0
    assert plan["execution_overrides"] == []
    assert inputs.physics_receipt["output_cadence"][
        "expected_frame_count"] == 2


@pytest.mark.parametrize(
    "profile",
    (
        runner.THOMPSON_PHYSICS_PROFILE,
        runner.MORRISON_PHYSICS_PROFILE,
        runner.NSSL2_PHYSICS_PROFILE,
        runner.NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
    ),
)
def test_20crv3_materialized_profiles_reach_exact_prepared_preflight(
        tmp_path, monkeypatch, profile):
    if profile == runner.THOMPSON_PHYSICS_PROFILE:
        _bind_test_thompson_runtime(tmp_path, monkeypatch)
    fixture = _prepared_fixture(
        tmp_path, "20crv3", physics_profile=profile)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)

    inputs = _preflight_fixture(fixture, physics_profile=profile)

    assert inputs.source == "20crv3"
    assert inputs.physics_receipt["profile"] == profile
    assert inputs.physics_receipt["resolved"]["mp_physics"] in {8, 10, 18}
    assert inputs.physics_receipt["validated_domains"][0]["grid_id"] == 1


def test_a_named_profile_still_binds_switch_for_switch():
    """A gate the caller asked for is not a gate the ruling dropped.

    Naming --physics-profile asserts the hash-bound experiment IS that
    shipped suite; a GFS experiment carrying different switches is still
    refused against it, with the drift named.  (This test used to pin
    the per-source scoping of the 20CRv3 profile; the scoping is gone --
    the mismatch refusal is what remains, and it is profile-vs-config,
    not profile-vs-source.)
    """

    exp = load_experiment(ROOT / "configs" / "gfs_wrf_direct_proof.toml")

    with pytest.raises(ValueError, match="experiment physics differs"):
        runner._validate_physics(
            exp, runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE,
            exp.run_seconds, 3600, source="gfs")


def test_the_product_default_suite_is_admitted_without_a_profile(
        tmp_path, monkeypatch):
    """POSITIVE CONTROL for the 2026-07-31 owner ruling.

    The exact suite ``gpuwm domain`` emits by default -- Thompson mp8,
    RTE+RRTMGP both streams, Kain-Fritsch, YSU/MM5/Noah -- was refused
    by this runner's whitelist against every named profile (watched
    firing below, and live in the field).  With no --physics-profile it
    is now admitted, its verification status is reported rather than
    gating, and the Thompson table couplings re-keyed on mp_physics == 8
    still bind.
    """

    from gpuwm.domain_wizard import DEFAULT_SUITE_PHYSICS

    _bind_test_thompson_runtime(tmp_path, monkeypatch)
    exp = load_experiment(ROOT / "configs" / "gfs_wrf_direct_proof.toml")
    run = replace(exp.root.run, **DEFAULT_SUITE_PHYSICS)
    root = replace(exp.root, run=run)
    exp = replace(exp, domains=(root,))

    # The refusal this control replaces, still firing for a NAMED
    # profile the config is not: the default suite is not the shipped
    # Thompson validation profile.
    with pytest.raises(ValueError, match="experiment physics differs"):
        runner._validate_physics(
            exp, runner.THOMPSON_PHYSICS_PROFILE, exp.run_seconds,
            float(exp.root.history_interval_s), source="gfs")

    receipt = runner._validate_physics(
        exp, None, exp.run_seconds,
        float(exp.root.history_interval_s), source="gfs")

    assert receipt["schema"] == "gpuwm-prepared-physics-suite-v1"
    assert receipt["profile"] is None
    assert receipt["profile_binding"] == "experiment-config"
    assert receipt["resolved"]["mp_physics"] == 8
    assert receipt["warning_only"] is True
    verification = receipt["verification"]
    assert verification["status"] == "supported-not-wrf-verified"
    assert "not yet WRF-verified" in verification["sentence"]
    assert "the run continues" in verification["sentence"]
    # The registry knows this tuple: the default template matches at
    # component level, reported without being claimed as switch-level
    # evidence.
    assert verification["component_matched_template"]["template"] \
        == "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1"
    # Coupling the removal must not drop: an mp8 suite admitted without
    # the Thompson profile id still binds the table authority and the
    # preflight-vs-run table-root stability check.
    assert "thompson_contract" in receipt
    assert runner._thompson_authority_paths(receipt)
    runner._verify_thompson_runtime_environment(receipt)


def test_an_unimplemented_mp_selector_still_refuses_naming_the_switch(
        tmp_path):
    """NEGATIVE CONTROL: suite freedom opens nothing the engine lacks.

    mp_physics = 95 (WRF's Ferrier-Aligo) has no kernel path in this
    tree.  (28 was the specimen until the aerosol-aware Thompson port
    made it real; a negative control must stay negative.)  The runner
    admits an experiment only through
    ``load_experiment`` -- preflight's first read of the hash-bound
    config -- and the engine validator refuses it there, naming the
    exact switch and the offending value.  The registry capability
    resolver refuses the same selector by name too.
    """

    # Not stripped: this control never materializes anything, it loads the
    # shipped config's own declared suite with one selector poisoned.
    text = (ROOT / "configs" / "gfs_wrf_direct_proof.toml").read_text(
        encoding="utf-8")
    assert text.count("mp_physics = 6") == 1
    bad = tmp_path / "unimplemented-mp.toml"
    bad.write_text(
        text.replace("mp_physics = 6", "mp_physics = 95"),
        encoding="utf-8")
    with pytest.raises(ValueError, match="mp_physics") as caught:
        load_experiment(bad)
    assert "95" in str(caught.value)

    from gpuwm.physics_compat import (
        PhysicsCapabilityError,
        validate_physics_capabilities,
    )
    selectors = runner.single_domain_runtime_switches(
        runner.WSM6_PROFILE_ID)
    selectors["mp_physics"] = 95
    with pytest.raises(PhysicsCapabilityError, match="mp_physics"):
        validate_physics_capabilities(selectors)


def test_unnamed_preflight_recomputes_the_front_door_selection(
        tmp_path, monkeypatch):
    """The profile-agnostic invariant of the route, watched at PREFLIGHT.

    The docstring of ``_validate_front_door_physics_proof`` calls the
    byte equality between the proof's per-domain selection receipt and
    the runner's recompute THE invariant; until this test nothing
    watched its ``MULTI_DOMAIN_SELECTION_SCHEMA`` branch fire through
    the production entrypoint.  The fixture's proof physics is written
    by the REAL GFS front door, JSON round trip included, and the
    negative control tampers one selector in the proof and watches the
    recompute refuse.
    """

    fixture = _prepared_fixture(tmp_path, "gfs", physics_profile=None)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)

    inputs = _preflight_fixture(fixture, physics_profile=None)

    proof_physics = inputs.proof["physics"]
    assert proof_physics["schema"] \
        == runner.MULTI_DOMAIN_SELECTION_SCHEMA
    assert proof_physics["profile"] is None
    assert inputs.proof["export"]["physics"] == proof_physics
    # The runner admitted the unnamed config on its own authority...
    assert inputs.physics_receipt["profile_binding"] in (
        "matched", "experiment-config")
    assert inputs.physics_receipt["registry_governance"][
        "state"] == "registry-reachable"

    # ...and the recompute REFUSES a proof whose recorded selection
    # differs from the hash-bound config, both spellings tampered
    # identically so this is the recompute firing, not the export
    # cross-check.
    proof = json.loads(fixture.proof.read_text(encoding="utf-8"))
    domain = proof["physics"]["domains"]["1"]
    assert domain["selectors"]["cu_physics"] != 1
    domain["selectors"]["cu_physics"] = 1
    proof["export"]["physics"] = proof["physics"]
    _write_json(fixture.proof, proof)
    with pytest.raises(
            ValueError,
            match="physics selection differs from the hash-bound"):
        _preflight_fixture(fixture, physics_profile=None)


def test_unnamed_forecast_main_reaches_execution_and_states_the_status(
        tmp_path, monkeypatch, capsys):
    """main() with NO --physics-profile: the production entrypoint.

    Everything up to and including preflight is real; only the GPU
    execution behind it is stubbed, because this suite runs where no
    CUDA device is granted.  The runner must admit the unnamed config,
    print the one-sentence verification status, and hand preflight's
    inputs to execution.
    """

    fixture = _prepared_fixture(tmp_path, "gfs", physics_profile=None)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    observed = {}

    # The double tracks the real signature, including `first_products`
    # -- the runner's own early render, which `main` now arms before the
    # preflight -- and `stream_init`, the initialization road a streamed
    # forecast takes.  A stub that swallowed **kwargs instead would keep
    # passing while the argument it was handed went nowhere.
    def fake_run(inputs, *, output_directory, observer=None,
                 first_products=None, stream_init="auto",
                 progress_options=None, preflight_seconds=None,
                 kernel_cache_census=None):
        observed["inputs"] = inputs
        observed["observer"] = observer
        observed["first_products"] = first_products
        observed["stream_init"] = stream_init
        observed["progress_options"] = progress_options
        observed["preflight_seconds"] = preflight_seconds
        observed["kernel_cache_census"] = kernel_cache_census
        return {
            "schema": runner.REPORT_SCHEMA,
            "status": "PASS",
            "source": inputs.source,
            "run_seconds": float(fixture.run_seconds),
            "history_interval_seconds": 3600.0,
            "gridded_output": {"frame_count": 2},
            "input": {
                "prepared_content_sha256": fixture.content_sha256},
        }

    monkeypatch.setattr(runner, "run_prepared_forecast", fake_run)
    rc = runner.main([
        "--source", "gfs",
        "--prepared-root", str(fixture.prepared),
        "--proof-sha256", _sha256(fixture.proof),
        "--source-manifest-sha256", _sha256(fixture.source_manifest),
        "--prepared-content-sha256", fixture.content_sha256,
        "--experiment-config", str(fixture.experiment),
        "--wps-namelist", str(fixture.wps),
        "--io-mode", "history",
        "--outdir", str(tmp_path / "outdir"),
    ])
    captured = capsys.readouterr()

    assert rc == 0
    assert "supported, not yet WRF-verified" in captured.err
    summary = json.loads(captured.out)
    assert summary["status"] == "PASS"
    receipt = observed["inputs"].physics_receipt
    assert receipt["profile_binding"] in ("matched", "experiment-config")
    assert receipt["verification"]["status"] == "supported-not-wrf-verified"
    # The preflight this door just ran is handed on as a number, not
    # left dark: the runner is the only place that can put it in the
    # step log and in report.json, and it did not measure it itself.
    assert isinstance(observed["preflight_seconds"], float)
    assert observed["preflight_seconds"] >= 0.0
    # The kernel cache as this run INHERITED it, read before the door
    # did anything: taken later it describes a cache this run has been
    # filling, which is how a card swap stayed silent.
    entries, undecodable, architectures = observed["kernel_cache_census"]
    assert isinstance(entries, int)
    assert isinstance(architectures, dict)


def test_materializer_unnamed_publishes_the_base_suite_unchanged(tmp_path):
    """--materialize-authorities with no profile: bytes are the product."""

    base = ROOT / "configs" / "gfs_wrf_direct_proof.toml"
    wps = ROOT / "configs" / "gfs_wrf_direct_proof.namelist.wps"
    output = tmp_path / "unnamed-authorities"

    receipt = runner.materialize_named_source_authorities(
        source="gfs", base_experiment_config=base,
        base_wps_namelist=wps, physics_profile=None,
        output_directory=output)

    assert receipt["status"] == "PASS"
    assert receipt["physics_profile"] is None
    assert receipt["readiness"] == "EXPERIMENT_CONFIG_SUITE"
    assert (output / "experiment.toml").read_bytes() == base.read_bytes()
    assert (output / "namelist.wps").read_bytes() == wps.read_bytes()
    validation = receipt["normalized_selected_physics"]
    assert validation["profile_binding"] == "experiment-config"
    assert validation["verification"]["schema"] \
        == "gpuwm-physics-verification-status-v1"


def test_matched_expert_suite_demands_the_registry_ack_at_the_runner(
        capsys):
    """The governance bypass the 2026-07-31 review found, now closed.

    A Noah-MP tuple that MATCHES the expert template is governed on the
    prepared ERA5 lane by the registry-owned acknowledgement -- exactly
    the consent the GFS front door records for the identical tuple --
    delivered by either published spelling.

    Unacknowledged, the tuple STILL RUNS and says so in one line naming
    both spellings, and the receipt carries ``acknowledged=False``: the
    warn-not-block ruling owns the severity of this site (an expert
    tuple is implemented and individually verified; what is missing is
    the registry's end-to-end blessing, which is a thing to state, not
    a thing to stop for).  What suite freedom owns, and what this test
    exists for, is that the governance is APPLIED here at all -- on the
    unnamed and matched paths, on every prepared source -- and that
    both delivery channels flip it, with provenance.  An explicitly
    NAMED expert profile is a gate the caller asked for and still
    refuses at the front door
    (``test_gfs_direct.py``); that is a different site with a different
    posture on purpose.
    """

    text = _asking_for_no_physics(
        (ROOT / "configs" / "gfs_wrf_direct_proof.toml").read_text(
            encoding="utf-8"))
    _rendered, exp, _receipt = runner._render_materialized_experiment(
        text, source="era5", profile=runner.NOAHMP_PHYSICS_PROFILE)

    unacked = runner._validate_physics(
        exp, None, exp.run_seconds, 3600, source="era5")
    message = capsys.readouterr().err
    warnings = [line for line in message.splitlines()
                if line.startswith("warning:")]
    # One line, not a paragraph, and it names the tuple's state and both
    # published ways to acknowledge it -- the same three facts the
    # refusal used to carry.
    assert len(warnings) == 1, warnings
    assert "registry-expert-template" in warnings[0]
    assert "--ack noahmp-host-column-throughput-v1" in warnings[0]
    assert 'acknowledgements = ["noahmp-host-column-throughput-v1"]' \
        in warnings[0]
    # Warning, not silence: the receipt states the unblessed truth.
    unacked_governance = unacked["registry_governance"]
    assert unacked_governance["state"] == "registry-expert-template"
    assert unacked_governance["acknowledged"] is False

    # TOML delivery (the hash-bound experiment's own array).
    acked = replace(
        exp, acknowledgements=("noahmp-host-column-throughput-v1",))
    receipt = runner._validate_physics(
        acked, None, acked.run_seconds, 3600, source="era5")
    assert receipt["profile_binding"] == "matched"
    governance = receipt["registry_governance"]
    assert governance["state"] == "registry-expert-template"
    assert governance["acknowledged"] is True
    assert governance["acknowledgement_provenance"] == {
        "noahmp-host-column-throughput-v1": [
            "[experiment].acknowledgements"]}
    # Acknowledging it is what silences the line.
    assert "registry-expert-template" not in capsys.readouterr().err

    # Flag delivery (the runner's own --ack).  The shipped proof config
    # this experiment materializes from also declares its nocturnal
    # asymmetric-radiation run (1.7.1), and the receipt records every
    # delivered declaration with its own provenance.
    receipt = runner._validate_physics(
        exp, None, exp.run_seconds, 3600, source="era5",
        expert_acknowledgements=("noahmp-host-column-throughput-v1",))
    assert receipt["registry_governance"]["acknowledged"] is True
    assert receipt["registry_governance"]["acknowledgement_provenance"] == {
        # Both radiation declarations ride the materialized experiment
        # since the constant-GLW guard: the profile runs Noah with lw 0
        # across a window with night in it, which is two separate claims
        # (the window, and the fabricated flux) and therefore two tokens.
        "asymmetric-radiation-nocturnal-window-v1": [
            "[experiment].acknowledgements"],
        "constant-downward-longwave-v1": [
            "[experiment].acknowledgements"],
        "noahmp-host-column-throughput-v1": ["--ack"]}
    assert "registry-expert-template" not in capsys.readouterr().err


def test_shipped_ruc_profile_is_not_governed_as_an_outside_tuple():
    """The single-domain union includes THIS route's own declarations.

    The registry declares RUC among the prepared single-domain route's
    ERA5 products; the domain-tree union alone would misread that
    shipped tuple as outside reachability and demand the authority
    acknowledgement for it.
    """

    text = _asking_for_no_physics(
        (ROOT / "configs" / "gfs_wrf_direct_proof.toml").read_text(
            encoding="utf-8"))
    _rendered, exp, _receipt = runner._render_materialized_experiment(
        text, source="era5", profile=runner.RUC_PHYSICS_PROFILE)

    receipt = runner._validate_physics(
        exp, None, exp.run_seconds, 3600, source="era5")

    assert receipt["profile_binding"] == "matched"
    assert receipt["registry_governance"]["state"] == "registry-reachable"
    assert receipt["registry_governance"]["required_acknowledgement"] is None


def test_kessler_readiness_reports_its_registry_maturity():
    """A named (or matched) Kessler must not read as more mature than
    the verification block beside it in the same receipt."""

    text = _asking_for_no_physics(
        (ROOT / "configs" / "gfs_wrf_direct_proof.toml").read_text(
            encoding="utf-8"))
    _rendered, exp, _receipt = runner._render_materialized_experiment(
        text, source="gfs", profile=runner.KESSLER_PHYSICS_PROFILE)

    named = runner._validate_physics(
        exp, runner.KESSLER_PHYSICS_PROFILE, exp.run_seconds, 3600,
        source="gfs")
    assert named["readiness"] == "IMPLEMENTED_UNVERIFIED"
    assert named["warning"] is not None
    assert named["verification"]["status"] == "supported-not-wrf-verified"

    matched = runner._validate_physics(
        exp, None, exp.run_seconds, 3600, source="gfs")
    assert matched["profile_binding"] == "matched"
    assert matched["readiness"] == "IMPLEMENTED_UNVERIFIED"

    # And the advertised catalog no longer omits the name it accepts.
    assert runner.KESSLER_PHYSICS_PROFILE in runner.PHYSICS_PROFILES
    capabilities = runner.runner_capabilities()
    assert runner.KESSLER_PHYSICS_PROFILE \
        in capabilities["physics_profiles"]


def test_cross_source_profile_receipt_records_the_run_source():
    """The receipt's source is the run's provenance, not the voucher's.

    Naming the 20CRv3-vouched profile on a GFS run used to stamp
    '20crv3' into the receipt -- a wrong label on exactly the
    cross-source admission the 2026-07-31 ruling opened.
    """

    text = _asking_for_no_physics(
        (ROOT / "configs" / "gfs_wrf_direct_proof.toml").read_text(
            encoding="utf-8"))
    _rendered, exp, _receipt = runner._render_materialized_experiment(
        text, source="gfs",
        profile=runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE)

    receipt = runner._validate_physics(
        exp, runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE, exp.run_seconds,
        3600, source="gfs")

    assert receipt["source"] == "gfs"
    assert receipt["readiness"] == "IMPLEMENTED_UNVERIFIED"


def test_20crv3_cache_identity_allows_only_the_legacy_same_scheme_default(
        tmp_path):
    fixture = _prepared_fixture(tmp_path, "20crv3")
    header = json.loads(
        (fixture.domain_bundle / "prepared-cache" / "header.json").read_text(
            encoding="utf-8"))
    expected = header["identity"]
    observed = json.loads(_canonical(expected))
    assert observed["domain_config"]["run"].pop(
        "nest_microphysics_transition") == "same-scheme-only"

    selected, receipt = runner._resolve_cache_identity_compatibility(
        source="20crv3", observed=observed, expected=expected)

    assert selected == observed
    assert receipt["status"] == "COMPATIBLE_LEGACY_DEFAULT"
    assert receipt["compatibility_overrides"][0]["field"] \
        == "domain_config.run.nest_microphysics_transition"
    observed["domain_config"]["run"]["mp_physics"] = 10
    with pytest.raises(ValueError, match="cache identity differs"):
        runner._resolve_cache_identity_compatibility(
            source="20crv3", observed=observed, expected=expected)


def test_preflight_accepts_exact_member_20crv3_mapped_hierarchy_d01(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "20crv3", hierarchy=True)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=True)

    inputs = _preflight_fixture(fixture)

    assert inputs.layout == "mapped-hierarchy-d01-v1"
    assert inputs.domain_bundle_path == fixture.domain_bundle.resolve()
    assert inputs.source_domain_count > 1
    assert inputs.source_member == "072"
    assert inputs.export_source_receipt["mp_physics"] == 6
    runner._verify_inputs_unchanged(inputs)


def _proof_content_sha(fixture) -> str:
    """The field INSIDE proof.json, which is not the file's digest."""

    proof = json.loads(fixture.proof.read_text(encoding="utf-8"))
    content = proof["proof_content_sha256"]
    assert content != _sha256(fixture.proof), (
        "instrument blind: the content field and the file digest are "
        "equal in this fixture, so the confusion cannot be reproduced")
    return content


def test_the_proof_content_field_passed_as_proof_sha256_is_named(tmp_path):
    """The digest confusion the mapped route walks a reader into.

    ``--proof-sha256`` wants the sha256 of the FILE ``proof.json``.  The
    document carries a field spelled ``proof_content_sha256``, and a
    reader with no printed command to paste finds that field, passes it,
    and gets `ValueError: preparation proof SHA differs from
    --proof-sha256` -- one sentence that names neither which digest is
    wanted, nor what was given, nor that the value they read out of the
    file can never match by construction.
    """

    fixture = _prepared_fixture(tmp_path, "20crv3")
    content = _proof_content_sha(fixture)

    with pytest.raises(runner.PreparationProofDigestMismatch) as refused:
        _preflight_fixture_with_proof_sha(fixture, content)

    message = str(refused.value)
    assert "proof_content_sha256" in message
    assert _sha256(fixture.proof) in message and content in message
    assert "sha256 of the file" in message


def test_a_digest_that_is_neither_says_it_is_neither(tmp_path):
    """The other arm: a stale or mistyped digest is not the content one."""

    fixture = _prepared_fixture(tmp_path, "20crv3")
    _proof_content_sha(fixture)  # the fixture really carries the field

    with pytest.raises(runner.PreparationProofDigestMismatch) as refused:
        _preflight_fixture_with_proof_sha(fixture, "0" * 64)

    message = str(refused.value)
    assert "neither" in message
    assert _sha256(fixture.proof) in message and "0" * 64 in message


def test_the_forecast_door_refuses_the_content_sha_without_a_traceback(
        tmp_path, monkeypatch, capsys):
    """Through main(): a refusal, an exit code, and no work started.

    A traceback where a user can land is a defect, and this one landed
    after the door had already claimed an output directory.
    """

    fixture = _prepared_fixture(tmp_path, "20crv3", physics_profile=None)
    content = _proof_content_sha(fixture)

    def never(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("execution started on a refused preparation")

    monkeypatch.setattr(runner, "run_prepared_forecast", never)
    outdir = tmp_path / "outdir"
    rc = runner.main([
        "--source", "20crv3",
        "--prepared-root", str(fixture.prepared),
        "--proof-sha256", content,
        "--source-manifest-sha256", _sha256(fixture.source_manifest),
        "--prepared-content-sha256", fixture.content_sha256,
        "--experiment-config", str(fixture.experiment),
        "--wps-namelist", str(fixture.wps),
        "--io-mode", "history",
        "--outdir", str(outdir),
    ])
    captured = capsys.readouterr()

    assert rc == 2
    assert "Traceback" not in captured.err
    assert "--proof-sha256 refused" in captured.err
    assert "proof_content_sha256" in captured.err
    assert _sha256(fixture.proof) in captured.err and content in captured.err
    # Refused before any directory was claimed: a mistyped digest must
    # not leave a run directory behind to clean up.
    assert not outdir.exists()


def test_20crv3_preflight_rejects_stale_mapped_proof_content_hash(
        tmp_path):
    fixture = _prepared_fixture(tmp_path, "20crv3")
    proof = json.loads(fixture.proof.read_text(encoding="utf-8"))
    proof["timing_seconds"]["total"] = 2.0
    _write_json(fixture.proof, proof)

    with pytest.raises(ValueError, match="proof content hash is stale"):
        _preflight_fixture(fixture)


def test_20crv3_preflight_rejects_manifest_cadence_drift(tmp_path):
    fixture = _prepared_fixture(tmp_path, "20crv3")
    manifest = json.loads(fixture.source_manifest.read_text(encoding="utf-8"))
    manifest["cadence_seconds"] = 3600
    content = dict(manifest)
    content.pop("content_sha256")
    manifest["content_sha256"] = hashlib.sha256(
        _canonical(content).encode()).hexdigest()
    _write_json(fixture.source_manifest, manifest)

    with pytest.raises(ValueError, match="manifest cadence"):
        _preflight_fixture(fixture)


def test_20crv3_preflight_rejects_member_swap_even_with_resealed_proof(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "20crv3")
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    manifest = json.loads(fixture.source_manifest.read_text(encoding="utf-8"))
    manifest["member"] = "071"
    for row in manifest["files"]:
        row["member"] = "071"
        row["filename"] = row["filename"].replace("mem072_", "mem071_", 1)
        row["path"] = str(Path(row["path"]).with_name(row["filename"]))
    content = dict(manifest)
    content.pop("content_sha256")
    manifest["content_sha256"] = hashlib.sha256(
        _canonical(content).encode()).hexdigest()
    _write_json(fixture.source_manifest, manifest)
    manifest_sha256 = _sha256(fixture.source_manifest)

    proof = json.loads(fixture.proof.read_text(encoding="utf-8"))
    composition = proof["source_composition"]
    # The receipt's input_manifest record deliberately stays the sealed
    # BRIDGED digest: the tamper is maximal everywhere the attacker can
    # reach without re-running a composition, and the refusal below must
    # come from the cache identity, not from a hash left stale.
    composition["alignment"]["member"] = "071"
    composition["terrain_products"] = [
        {"path": row["path"], "sha256": row["sha256"]}
        for row in manifest["files"] if row["role"] == "sfc"
    ]
    composition.pop("receipt_content_sha256")
    composition["receipt_content_sha256"] = hashlib.sha256(
        _canonical(composition).encode()).hexdigest()
    proof.pop("proof_content_sha256")
    proof["proof_content_sha256"] = hashlib.sha256(
        _canonical(proof).encode()).hexdigest()
    _write_json(fixture.proof, proof)

    with pytest.raises(ValueError, match="source identity differs"):
        _preflight_fixture(fixture)


def _reseal_mapped_proof(fixture) -> None:
    """Re-hash the composition receipt and the proof, in that order.

    A tamper test that leaves a stale hash behind proves only that the
    hash check works.  Re-sealing both is what makes the refusal that
    fires the one the test is actually about.
    """

    proof = json.loads(fixture.proof.read_text(encoding="utf-8"))
    composition = proof["source_composition"]
    composition.pop("receipt_content_sha256", None)
    composition["receipt_content_sha256"] = hashlib.sha256(
        _canonical(composition).encode()).hexdigest()
    proof.pop("proof_content_sha256", None)
    proof["proof_content_sha256"] = hashlib.sha256(
        _canonical(proof).encode()).hexdigest()
    _write_json(fixture.proof, proof)


def test_source_20crv3_refuses_a_bundle_prepared_from_a_users_own_mapping(
        tmp_path, monkeypatch):
    """THE control on de-specialising the mapped route.

    ``20crv3`` is the declarative mapped route wearing a specific name:
    it writes the same proof schema, the same ``source-evidence``
    directory and the same composition receipt as any other mapped
    preparation.  What makes it ``20crv3`` and not "some mapping" is the
    certificate -- the mapping, composition and provenance authorities
    must hash to the ones PACKAGED with this distribution.

    That is exactly the property that must survive opening the route up
    to caller-supplied mappings.  Widening the certificate instead of
    adding a second, narrower one would turn a specific route into a
    permissive one: any bundle at all would then be accepted as 20CRv3,
    and the receipts would say 20CRv3 about data that never came from
    it.

    Here the bundle is well formed and internally consistent -- every
    digest re-sealed, so nothing else can fire -- and differs from the
    packaged route in exactly one way: the mapping authority is the
    caller's.  ``--source 20crv3`` must refuse it, naming the packaged
    authorities.
    """

    fixture = _prepared_fixture(tmp_path, "20crv3")
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    # Sanity, both directions: the untouched bundle is accepted, so a
    # refusal below is caused by the substitution and by nothing else.
    runner._verify_inputs_unchanged(_preflight_fixture(fixture))

    evidence = fixture.proof.parent / "source-evidence"
    users_mapping = json.loads(
        (evidence / "mapping.json").read_text(encoding="utf-8"))
    users_mapping["name"] = "a-mapping-this-user-authored"
    _write_json(evidence / "mapping.json", users_mapping)
    users_digest = _sha256(evidence / "mapping.json")
    assert users_digest != runner.twentycrv3_authority_sha256()["mapping"]

    proof = json.loads(fixture.proof.read_text(encoding="utf-8"))
    proof["source_composition"]["mapping"]["sha256"] = users_digest
    _write_json(fixture.proof, proof)
    _reseal_mapped_proof(fixture)

    with pytest.raises(
        ValueError, match="does not use the packaged 20crv3 authorities"
    ):
        _preflight_fixture(fixture)


def test_20crv3_preflight_and_execution_bind_copied_provenance(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "20crv3")
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    inputs = _preflight_fixture(fixture)
    inputs.authority_paths["mapped_provenance"].write_bytes(b"drift")

    with pytest.raises(RuntimeError, match="inputs changed during execution"):
        runner._verify_inputs_unchanged(inputs)


def test_materialized_wsm6_physics_receipt_is_canonical():
    text = _asking_for_no_physics(
        (ROOT / "configs" / "gfs_wrf_direct_proof.toml").read_text(
            encoding="utf-8"))
    _rendered, exp, _materialization = (
        runner._render_materialized_experiment(
            text, source="gfs", profile=runner.PHYSICS_PROFILE))
    receipt = runner._validate_physics(
        exp, runner.PHYSICS_PROFILE, exp.run_seconds, 3600, source="gfs")

    assert receipt["resolved"] == {
        **runner.single_domain_runtime_switches(runner.PHYSICS_PROFILE),
        "radiation_scheme_ids": [0, 1],
    }


# RUC is absent from this GFS parametrization on purpose.  It used to be
# here, and it passed: a GFS-initialised RUC bundle really does reach the
# v3 preflight.  What it cannot do is complete the first step of the
# forecast that preflight clears it for -- `mavail must be finite`, at
# model_elapsed_seconds 0.0 -- so v1.1.1 withdrew the pairing at the
# front door rather than keep clearing a run that ends in a guard.  RUC
# on ERA5 is exercised by test_era5_ruc_family_reaches_prepared_v3_preflight
# below; the two together are the whole of what is claimed for it.
@pytest.mark.parametrize(
    ("profile", "sfclay", "surface", "pbl", "soil_layers"),
    (
        (runner.MYNN_PHYSICS_PROFILE, 5, 2, 5, 4),
        (runner.NOAHMP_PHYSICS_PROFILE, 91, 4, 1, 4),
    ),
)
def test_gfs_new_front_door_families_reach_prepared_v3_preflight(
        tmp_path, monkeypatch, capsys, profile, sfclay, surface, pbl,
        soil_layers):
    fixture = _prepared_fixture(
        tmp_path, "gfs", physics_profile=profile)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)

    expert = profile == runner.NOAHMP_PHYSICS_PROFILE
    acknowledgements = (
        ("noahmp-host-column-throughput-v1",) if expert else ())
    if expert:
        # The registry-owned expert acknowledgement is applied AT THE
        # RUNNER too (its governance is source-neutral tuple
        # governance, the same one the front doors apply), and it names
        # both delivery spellings.  Severity here is warn-not-block's:
        # the tuple runs unblessed and says so in one line.
        _preflight_fixture(fixture, physics_profile=profile)
        unacked = [line for line in capsys.readouterr().err.splitlines()
                   if line.startswith("warning:")
                   and "registry-expert-template" in line]
        assert len(unacked) == 1, unacked
        assert "--ack noahmp-host-column-throughput-v1" in unacked[0]
        assert 'acknowledgements = ["noahmp-host-column-throughput-v1"]' \
            in unacked[0]

    inputs = _preflight_fixture(
        fixture, physics_profile=profile,
        expert_acknowledgements=acknowledgements)

    assert inputs.proof["schema"] == "gpuwm-gfs-direct-wrf-proof-v3"
    assert inputs.proof["physics"]["profile"] == profile
    assert inputs.proof["export"]["schema"] \
        == "gpuwm-native-direct-wrf-export-v3"
    assert inputs.proof["export"]["physics"] == inputs.proof["physics"]
    governance = inputs.physics_receipt["registry_governance"]
    assert governance["acknowledged"] is True
    assert governance["state"] == (
        "registry-expert-template" if expert else "registry-reachable")
    resolved = inputs.physics_receipt["resolved"]
    assert resolved["sf_sfclay_physics"] == sfclay
    assert resolved["sf_surface_physics"] == surface
    assert resolved["bl_pbl_physics"] == pbl
    assert resolved["num_soil_layers"] == soil_layers


def test_era5_ruc_family_reaches_prepared_v3_preflight(tmp_path, monkeypatch):
    """RUC is withdrawn from GFS only, and this is the half that proves it.

    A withdrawal that quietly took the component out of the product
    would pass every test that merely stopped asserting it.  ERA5 still
    offers RUC, nobody has shown that pairing to be broken, and it must
    keep reaching the same v3 preflight the GFS one no longer does.
    """
    profile = runner.RUC_PHYSICS_PROFILE
    fixture = _prepared_fixture(tmp_path, "era5", physics_profile=profile)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)

    inputs = _preflight_fixture(fixture, physics_profile=profile)
    resolved = inputs.physics_receipt["resolved"]
    assert resolved["sf_surface_physics"] == 3
    assert resolved["num_soil_layers"] == 9


def test_preflight_accepts_exact_nssl2_validation_candidate_and_binds_receipt(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(
        tmp_path, "gfs", physics_profile=runner.NSSL2_PHYSICS_PROFILE)
    grid = SimpleNamespace(source="gfs")
    monkeypatch.setattr(
        runner, "validate_native_lambert_contract",
        lambda exp, path, *, source_name: grid)
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {"STATIC": np.ones((ny, nx))})

    inputs = runner.preflight_prepared_forecast(
        source="gfs", prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.NSSL2_PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds, history_interval_seconds=3600)

    receipt = inputs.physics_receipt
    assert receipt["profile"] == runner.NSSL2_PHYSICS_PROFILE
    assert receipt["readiness"] == "WRF_MATCHED_RUN_CANDIDATE"
    assert receipt["resolved"]["mp_physics"] == 18
    assert receipt["resolved"]["moist"] is True
    assert receipt["resolved"]["moist_cq"] is True
    assert receipt["resolved"]["epssm"] == 0.5
    assert receipt["resolved"]["radiation_scheme_ids"] == [4, 4]
    assert receipt["nssl2_contract"]["contract_id"] == (
        "wrf-v4.6.1-nssl-mp18-two-moment-hail-ccn-density-v1"
    )
    assert receipt["nssl2_contract"]["selector"] == 18
    assert receipt["nssl2_contract"]["is_default_lane"] is True
    assert receipt["nssl2_contract"]["absent_fields"] == []
    assert receipt["nssl2_contract"]["resolved_mode"] == {
        "two_moment": True,
        "hail": True,
        "predicted_ccn": True,
        "density_moments": 2,
        "sixth_moments": 0,
    }
    assert receipt["radiation_substitution"] == {
        "contract": "wrf-rrtmg-4-4-to-rte-rrtmgp-v2",
        "requested_wrf_scheme_ids": [4, 4],
        "resolved_gpuwm_scheme_ids": [4, 4],
        "resolved_gpuwm_solver": "RTE+RRTMGP",
    }


def test_nssl2_profile_rejects_wsm6_experiment_before_cache_restore(tmp_path):
    fixture = _prepared_fixture(tmp_path, "gfs")

    with pytest.raises(ValueError, match="NSSL-2 wrf-matched-run-candidate"):
        runner.preflight_prepared_forecast(
            source="gfs", prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.NSSL2_PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600)


@pytest.mark.parametrize("source", ("gfs", "era5"))
def test_preflight_accepts_materialized_morrison_for_named_sources(
        tmp_path, monkeypatch, source):
    fixture = _prepared_fixture(
        tmp_path, source, physics_profile=runner.MORRISON_PHYSICS_PROFILE)
    grid = SimpleNamespace(source=source)
    monkeypatch.setattr(
        runner, "validate_native_lambert_contract",
        lambda exp, path, *, source_name: grid)
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {"STATIC": np.ones((ny, nx))})

    inputs = runner.preflight_prepared_forecast(
        source=source, prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.MORRISON_PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds, history_interval_seconds=3600)

    receipt = inputs.physics_receipt
    assert receipt["profile"] == runner.MORRISON_PHYSICS_PROFILE
    assert receipt["readiness"] == "WRF_MATCHED_RUN_RUNTIME_PROFILE"
    assert receipt["resolved"]["mp_physics"] == 10
    assert receipt["resolved"]["morr_rimed_ice"] == 1
    assert receipt["resolved"]["radiation_scheme_ids"] == [4, 4]
    assert receipt["morrison_contract"]["rimed_ice_category"] == "hail"


@pytest.mark.parametrize("source", ("gfs", "era5"))
def test_preflight_accepts_guarded_thompson_for_each_prepared_source(
        tmp_path, monkeypatch, source):
    runtime = _bind_test_thompson_runtime(tmp_path, monkeypatch)
    fixture = _prepared_fixture(
        tmp_path, source, physics_profile=runner.THOMPSON_PHYSICS_PROFILE)
    grid = SimpleNamespace(source=source)
    monkeypatch.setattr(
        runner, "validate_native_lambert_contract",
        lambda exp, path, *, source_name: grid)
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {"STATIC": np.ones((ny, nx))})

    inputs = runner.preflight_prepared_forecast(
        source=source, prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.THOMPSON_PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds, history_interval_seconds=3600)

    receipt = inputs.physics_receipt
    assert receipt["profile"] == runner.THOMPSON_PHYSICS_PROFILE
    assert receipt["readiness"] == "WRF_MATCHED_RUN_EXPERIMENTAL_RUNTIME"
    assert receipt["resolved"]["mp_physics"] == 8
    assert receipt["resolved"]["moist"] is True
    assert receipt["resolved"]["moist_cq"] is True
    assert receipt["resolved"]["epssm"] == 0.5
    assert receipt["resolved"]["radiation_scheme_ids"] == [0, 1]
    contract = receipt["thompson_contract"]
    assert contract["selector"] == 8
    assert contract["transported_fields"] == [
        "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr"]
    assert contract["native_reflectivity"] == {
        "field": "REFL_10CM",
        "producer": "classic-Thompson same-call graupel-number shadow",
        "consumer": "gpuwm.core.refl.consume_refl_10cm",
        "fallback": None,
    }
    table = contract["table_authority"]
    assert table["root"] == str(runtime.root)
    assert table["payload_bytes"] == 379_839_912
    assert table["assets"] == [
        {"filename": asset.filename, "bytes": asset.bytes,
         "sha256": asset.sha256}
        for asset in runtime.assets
    ]
    table_identity = {
        key: table[key]
        for key in ("schema", "table_set", "wrf_version", "wrf_commit", "assets")
    }
    assert table["identity_sha256"] == hashlib.sha256(
        _canonical(table_identity).encode("ascii")).hexdigest()
    assert len(contract["implementation_sha256"]) == len(
        runner._THOMPSON_IMPLEMENTATION_FILES)
    assert len([
        name for name in inputs.file_sha256
        if name.startswith("thompson_table_")
    ]) == len(runtime.assets)


def test_thompson_profile_rejects_wsm6_cache_before_restore(
        tmp_path, monkeypatch):
    _bind_test_thompson_runtime(tmp_path, monkeypatch)
    fixture = _prepared_fixture(tmp_path, "gfs")

    with pytest.raises(ValueError, match="guarded Thompson MP8"):
        runner.preflight_prepared_forecast(
            source="gfs", prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.THOMPSON_PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600)


def test_wsm6_profile_rejects_thompson_cache_before_restore(
        tmp_path, monkeypatch):
    _bind_test_thompson_runtime(tmp_path, monkeypatch)
    fixture = _prepared_fixture(
        tmp_path, "gfs", physics_profile=runner.THOMPSON_PHYSICS_PROFILE)

    with pytest.raises(ValueError, match="supported prepared-cache profile"):
        runner.preflight_prepared_forecast(
            source="gfs", prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600)


def test_thompson_runs_without_the_retired_enable_gate(tmp_path, monkeypatch):
    """mp8 no longer demands GPUWM_EXPERIMENTAL_THOMPSON_MP8=1.

    The gate predated the packaging promotion: the classic tables ship
    as package data and `gpuwm doctor` byte-validates all four.  Keeping
    it meant the wizard's own default microphysics failed twice at
    runtime on a machine doctor had just called clean, with neither
    variable named in any document.  The table-root binding, which is
    what actually protected anything, is unchanged.
    """
    runtime = _bind_test_thompson_runtime(tmp_path, monkeypatch)
    monkeypatch.delenv("GPUWM_EXPERIMENTAL_THOMPSON_MP8", raising=False)
    fixture = _prepared_fixture(
        tmp_path, "gfs", physics_profile=runner.THOMPSON_PHYSICS_PROFILE)
    exp = load_experiment(fixture.experiment)
    receipt = runner._validate_physics(
        exp, runner.THOMPSON_PHYSICS_PROFILE, exp.run_seconds, 3600)
    guard = receipt["thompson_contract"]["guard"]
    assert "experimental_runtime_environment" not in guard
    assert guard["table_root_source"] == "environment override"
    # And with no override at all, a COMPLETE packaged directory
    # resolves.  Both roots are injected rather than read off this box:
    # the packaged set is complete in a git checkout and in a wheel whose
    # user has run `gpuwm fetch-tables`, and short two 300 MiB assets in
    # every other wheel -- so an uninjected assertion here pins whichever
    # of the two the machine running the suite happens to have, which is
    # exactly how this test passed on Windows and failed on Linux at the
    # same commit.  What the product promises is the ORDER, and the order
    # is what is pinned below.
    monkeypatch.delenv(runner.THOMPSON_TABLE_ROOT_ENV)
    from gpuwm import physics_compat
    from gpuwm.core.thompson_contract import CLASSIC_TABLE_ASSETS

    def _table_root(where, *, complete):
        where.mkdir(parents=True, exist_ok=True)
        assets = (CLASSIC_TABLE_ASSETS if complete
                  else CLASSIC_TABLE_ASSETS[:-1])
        for asset in assets:
            (where / asset.filename).write_bytes(b"")
        return where

    packaged = _table_root(tmp_path / "packaged", complete=True)
    staged = _table_root(tmp_path / "staged", complete=True)
    monkeypatch.setattr(physics_compat, "packaged_thompson_table_root",
                        lambda: packaged)
    monkeypatch.setattr(physics_compat, "user_thompson_table_root",
                        lambda: staged)
    assert Path(physics_compat.thompson_table_root()) == packaged

    # A wheel whose packaged root is short an asset reads the staged set
    # instead -- the read fallback that keeps `gpuwm fetch-tables` worth
    # running -- rather than refusing a run that can read every table.
    short = _table_root(tmp_path / "short-packaged", complete=False)
    monkeypatch.setattr(physics_compat, "packaged_thompson_table_root",
                        lambda: short)
    assert Path(physics_compat.thompson_table_root()) == staged

    # Neither root complete: the companion's own refusal is what the
    # user gets, because "no tables anywhere" and "no companion" want
    # different fixes.
    monkeypatch.setattr(physics_compat, "user_thompson_table_root",
                        lambda: _table_root(tmp_path / "short-staged",
                                            complete=False))
    monkeypatch.setattr(
        physics_compat, "packaged_thompson_table_root",
        lambda: (_ for _ in ()).throw(ImportError("gpuwm-data is missing")))
    with pytest.raises(ImportError, match="gpuwm-data is missing"):
        physics_compat.thompson_table_root()

    monkeypatch.setenv(runner.THOMPSON_TABLE_ROOT_ENV, str(runtime.root))
    changed = tmp_path / "other-tables"
    changed.mkdir()
    monkeypatch.setenv(runner.THOMPSON_TABLE_ROOT_ENV, str(changed))
    with pytest.raises(RuntimeError,
                       match="changed between preflight and run"):
        runner._verify_thompson_runtime_environment(receipt)
    assert receipt["thompson_contract"]["table_authority"]["root"] == str(
        runtime.root)


def test_thompson_table_drift_is_rejected_by_preflight(tmp_path, monkeypatch):
    runtime = _bind_test_thompson_runtime(tmp_path, monkeypatch)
    fixture = _prepared_fixture(
        tmp_path, "era5", physics_profile=runner.THOMPSON_PHYSICS_PROFILE)
    (runtime.root / runtime.assets[0].filename).write_bytes(b"drift")

    with pytest.raises(ValueError, match="test Thompson table drift"):
        runner.preflight_prepared_forecast(
            source="era5", prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.THOMPSON_PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600)


def test_preflight_derives_hash_bound_hierarchy_d01_bundle(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "gfs", hierarchy=True)
    grid = SimpleNamespace(source="gfs")
    monkeypatch.setattr(
        runner, "validate_native_lambert_contracts",
        lambda exp, path, *, source_name: tuple(grid for _ in exp.domains))
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {"STATIC": np.ones((ny, nx))})

    inputs = runner.preflight_prepared_forecast(
        source="gfs", prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds, history_interval_seconds=3600)

    assert inputs.layout == "hierarchy-d01-v1"
    assert inputs.domain_bundle_path == fixture.domain_bundle.resolve()
    assert inputs.source_domain_count > 1
    assert inputs.experiment.domains == (inputs.experiment.root,)


def test_hierarchy_d01_nssl2_authority_uses_dynamic_microphysics_label(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(
        tmp_path, "gfs", hierarchy=True,
        physics_profile=runner.NSSL2_PHYSICS_PROFILE)
    grid = SimpleNamespace(source="gfs")
    monkeypatch.setattr(
        runner, "validate_native_lambert_contracts",
        lambda exp, path, *, source_name: tuple(grid for _ in exp.domains))
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {"STATIC": np.ones((ny, nx))})

    inputs = runner.preflight_prepared_forecast(
        source="gfs", prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.NSSL2_PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds, history_interval_seconds=3600)

    assert inputs.export_source_receipt["mp_physics"] == 18
    assert inputs.export_source_receipt["microphysics"] == "NSSL-2"


def test_preflight_rejects_explicit_hierarchy_bundle_outside_manifest(
        tmp_path):
    fixture = _prepared_fixture(tmp_path, "gfs", hierarchy=True)
    wrong = tmp_path / "other-d01"
    wrong.mkdir()

    with pytest.raises(ValueError, match="manifest-derived hierarchy d01"):
        runner.preflight_prepared_forecast(
            source="gfs", prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600,
            domain_bundle=wrong)


@pytest.mark.parametrize("source", ("gfs", "era5"))
def test_preflight_rejects_wrong_source_adapter_even_when_bundle_is_self_consistent(
        tmp_path, monkeypatch, source):
    fixture = _prepared_fixture(
        tmp_path, source, adapter="era5-grib1-direct-v1" if source == "gfs"
        else "gfs-pgrb2-0p25-direct-v1")
    monkeypatch.setattr(
        runner, "validate_native_lambert_contract", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "verify_native_static_receipt", lambda *args: {})
    monkeypatch.setattr(runner, "load_native_static_cache", lambda *args: {})

    with pytest.raises(ValueError, match="source identity differs"):
        runner.preflight_prepared_forecast(
            source=source, prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600)


def test_preflight_rejects_pinned_proof_drift_before_cache_restore(
        tmp_path):
    fixture = _prepared_fixture(tmp_path, "gfs")
    with pytest.raises(ValueError, match=r"--proof-sha256 refused"):
        runner.preflight_prepared_forecast(
            source="gfs", prepared_root=fixture.prepared,
            proof_sha256="0" * 64,
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600)


def test_gfs_v2_proof_remains_verifiable_as_its_own_schema(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "gfs")
    proof = json.loads(fixture.proof.read_text(encoding="utf-8"))
    proof["schema"] = "gpuwm-gfs-direct-wrf-proof-v2"
    proof.pop("physics")
    proof["export"]["schema"] = "gpuwm-native-direct-wrf-export-v2"
    proof["export"].pop("physics")
    _write_json(fixture.proof, proof)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)

    inputs = _preflight_fixture(fixture)

    assert inputs.proof["schema"] == "gpuwm-gfs-direct-wrf-proof-v2"
    assert "physics" not in inputs.proof


def test_gfs_v3_proof_requires_hash_bound_physics_selection(tmp_path):
    fixture = _prepared_fixture(tmp_path, "gfs")
    proof = json.loads(fixture.proof.read_text(encoding="utf-8"))
    proof.pop("physics")
    _write_json(fixture.proof, proof)

    with pytest.raises(ValueError, match="v3 preparation proof physics"):
        _preflight_fixture(fixture)


def test_gfs_v3_export_must_repeat_exact_preparation_physics(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "gfs")
    proof = json.loads(fixture.proof.read_text(encoding="utf-8"))
    proof["export"]["physics"]["profile"] = runner.MYNN_PHYSICS_PROFILE
    _write_json(fixture.proof, proof)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)

    with pytest.raises(ValueError, match="export physics receipt differs"):
        _preflight_fixture(fixture)


def test_gfs_unknown_future_proof_schema_fails_closed(tmp_path):
    fixture = _prepared_fixture(tmp_path, "gfs")
    proof = json.loads(fixture.proof.read_text(encoding="utf-8"))
    proof["schema"] = "gpuwm-gfs-direct-wrf-proof-v4"
    _write_json(fixture.proof, proof)

    with pytest.raises(ValueError, match="unsupported GFS preparation proof"):
        _preflight_fixture(fixture)


def test_preflight_rejects_resolved_wrf_contract_drift(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "era5")
    proof = json.loads(fixture.proof.read_text(encoding="utf-8"))
    proof["export"]["source"]["resolved_physics_contract_sha256"] = "0" * 64
    _write_json(fixture.proof, proof)
    monkeypatch.setattr(
        runner, "validate_native_lambert_contract", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "verify_native_static_receipt", lambda *args: {})
    monkeypatch.setattr(runner, "load_native_static_cache", lambda *args: {})

    with pytest.raises(ValueError, match="export source hashes differ"):
        runner.preflight_prepared_forecast(
            source="era5", prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600)


def test_output_claim_is_create_only_and_preserves_existing_tree(tmp_path):
    output = tmp_path / "run"
    assert runner.claim_output_directory(output) == output.resolve()
    marker = output / "owned-by-first-run.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        runner.claim_output_directory(output)

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_output_claim_accepts_the_empty_folder_the_stage_door_just_claimed(
        tmp_path):
    """`gpuwm sim` is the caller, and it allocates before it dispatches.

    The breakage this claim prevents is MERGING INTO AN EARLIER RUN --
    two runs' wrfout frames under one set of names and a report.json
    describing only the last.  An EMPTY directory holds no earlier run,
    so refusing it prevents nothing and costs the whole route.

    That is not hypothetical.  `gpuwm sim`'s run-folder claim is
    create-exclusive on purpose (two launches in one second must not
    share a folder), so by the time it dispatches, the stamped folder
    EXISTS and is empty.  Against the real artifact on weather-node-1,
    the merged 2.5.0 tip refused its own stamped folder and the forecast
    never started:

        prepared_single_domain_forecast: --outdir refused:
        .../run-20260818-073611Z_i202608180600Z already exists

    Both halves were individually right, which is why neither side's
    suite caught it: the door names a real breakage, the runner names a
    real breakage, and the composition ran nothing.
    """

    claimed = tmp_path / "fc" / "run-20260818-073611Z_i202608180600Z"
    claimed.mkdir(parents=True)

    assert runner.claim_output_directory(claimed) == claimed.resolve()

    # ...and the refusal still fires the moment it holds anything.
    (claimed / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already holds"):
        runner.claim_output_directory(claimed)


def test_output_claim_refuses_to_modify_the_prepared_input_tree(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()

    with pytest.raises(ValueError, match="protected input tree"):
        runner.claim_output_directory(
            prepared / "forecast", protected_roots=(prepared,))

    assert not (prepared / "forecast").exists()


def test_output_claim_delivers_an_unresolvable_path_as_a_sentence(tmp_path):
    """E-12: this function's whole subject is turning an OS-level
    exception into a sentence, and resolve() -- the line BEFORE its first
    refusal -- was the one it let through.

    A --outdir whose symlink points into its own chain raised OSError
    (Errno 40) straight past the caller's (ValueError, FileExistsError)
    handler, so it reached the reader as a traceback at rc 1.
    """
    import os

    loop = tmp_path / "loop"
    try:
        os.symlink(str(loop), str(loop))
    except (OSError, NotImplementedError, AttributeError) as error:
        # Unprivileged Windows cannot create one.  Skip rather than
        # weaken: the fleet that filed this runs Linux, where it runs.
        pytest.skip(f"cannot create a looping symlink here: {error}")
    with pytest.raises(ValueError, match="cannot be resolved") as raised:
        runner.claim_output_directory(loop, flag="--output-directory")
    message = str(raised.value)
    # the flag the reader actually typed, and the errno, both survive
    assert "--output-directory" in message
    # pathlib raises RuntimeError("Symlink loop from ...") here, not
    # OSError(ELOOP) -- an OSError-only guard misses it, which is what
    # the first version of this fix did.  Either cause is acceptable;
    # neither may escape as itself.
    assert isinstance(raised.value.__cause__, (OSError, RuntimeError))
    assert "cannot be resolved" in message


@pytest.mark.parametrize("failure", [
    RuntimeError("Symlink loop from '/x'"),
    OSError(40, "Too many levels of symbolic links"),
])
def test_output_claim_converts_either_resolve_failure(
        tmp_path, monkeypatch, failure):
    """Runs everywhere, including where the symlink test above skips.

    pathlib reports a symlink loop as RuntimeError on some paths and
    OSError(ELOOP) on others; a guard that knows only one of them leaves
    the other as a traceback.  Both are pinned because the platform that
    can create the real loop is not the platform this is written on.
    """
    def boom(self, *_a, **_k):
        raise failure

    monkeypatch.setattr(Path, "resolve", boom)
    with pytest.raises(ValueError, match="cannot be resolved") as raised:
        runner.claim_output_directory(tmp_path / "out")
    assert raised.value.__cause__ is failure


def test_durable_wrfout_inventory_reads_atomic_writer_subdirectory(tmp_path):
    output = tmp_path / "model-output"
    wrfout = output / "wrfout"
    wrfout.mkdir(parents=True)
    first = wrfout / "wrfout_d01_2026-07-21_12_00_00"
    second = wrfout / "wrfout_d01_2026-07-21_13_00_00"
    ignored = output / "wrfout_d01_wrong_location"
    first.write_bytes(b"first-complete-frame")
    second.write_bytes(b"second-complete-frame")
    ignored.write_bytes(b"must-not-be-inventoried")

    inventory = runner._durable_wrfout_inventory(output)

    assert inventory == [
        {
            "path": str(first.resolve()),
            "bytes": first.stat().st_size,
            "sha256": _sha256(first),
        },
        {
            "path": str(second.resolve()),
            "bytes": second.stat().st_size,
            "sha256": _sha256(second),
        },
    ]


def test_restored_source_adapter_hint_is_optional_but_never_conflicting():
    runner._validate_restored_source_adapter({}, "gfs")
    runner._validate_restored_source_adapter({"source_adapter": "gfs"}, "gfs")
    runner._validate_restored_source_adapter(
        {"source_adapter": "mapped"}, "20crv3")

    with pytest.raises(ValueError, match="source adapter differs"):
        runner._validate_restored_source_adapter(
            {"source_adapter": "era5"}, "gfs")
    with pytest.raises(ValueError, match="source adapter differs"):
        runner._validate_restored_source_adapter(
            {"source_adapter": "20crv3"}, "20crv3")


def test_disabled_optional_physics_records_zero_updates():
    assert runner._physics_update_count(None) == 0
    assert runner._physics_update_count(SimpleNamespace(update_count=7)) == 7

    with pytest.raises(ValueError, match="cannot be negative"):
        runner._physics_update_count(SimpleNamespace(update_count=-1))


def test_post_restore_content_pin_is_checked_before_forecast():
    expected = "a" * 64
    runner._validate_restored_cache_receipt({
        "schema": PREPARED_CACHE_SCHEMA,
        "status": "RESTORED",
        "content_sha256": expected,
    }, expected)

    with pytest.raises(ValueError, match="caller-pinned content"):
        runner._validate_restored_cache_receipt({
            "schema": PREPARED_CACHE_SCHEMA,
            "status": "RESTORED",
            "content_sha256": "b" * 64,
        }, expected)


def test_source_neutral_surface_contract_accepts_exact_noah_inventory():
    shape = (3, 4)
    fields = {
        name: np.ones((4, *shape) if name in {"TSLB", "SMOIS", "SH2O"}
                      else shape, dtype=np.float32)
        for name in runner._CANONICAL_SURFACE_FIELDS
    }
    fields["LANDMASK"][:] = 0.0
    fields["XLAND"][:] = 2.0

    # The prepared-surface contract is Noah's; the soil rank it enforces is
    # now resolved from the SCHEME (config.soil_layer_count), so the cfg
    # stand-in has to say which scheme it is rather than only how big.
    actual = _validate_prepared_surface(
        SimpleNamespace(fields=fields),
        SimpleNamespace(ny=3, nx=4, sf_surface_physics=2, num_soil_layers=4))

    assert actual is fields


def test_source_neutral_near_surface_contract_accepts_exact_staggering():
    cfg = SimpleNamespace(ny=3, nx=4)
    result = SimpleNamespace(
        surface_pressure=np.full((3, 4), 95_000.0),
        surface_qv=np.full((3, 4), 0.01),
    )
    met = SimpleNamespace(fields={
        "T2": np.full((3, 4), 290.0),
        "SKINTEMP": np.full((3, 4), 291.0),
        "U10": np.zeros((3, 5)),
        "V10": np.zeros((4, 4)),
        "LANDSEA": np.ones((3, 4)),
    })

    actual = _validate_prepared_near_surface(result, met, cfg)

    assert actual["surface_pressure"].shape == (3, 4)
    assert actual["U10"].shape == (3, 5)
    assert actual["V10"].shape == (4, 4)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("surface_pressure", np.asarray(95_000.0), "surface_pressure has shape"),
        ("surface_qv", np.full((2, 3), np.nan), "surface_qv must be finite"),
        ("T2", np.full((2, 3), 500.0), "T2 is outside the physical range"),
        ("U10", np.zeros((2, 3)), "U10 has shape"),
        ("LANDSEA", np.full((2, 3), 0.5), "LANDSEA must be exactly binary"),
    ],
)
def test_source_neutral_near_surface_contract_rejects_bad_arrays(
        field, value, error):
    cfg = SimpleNamespace(ny=2, nx=3)
    result = SimpleNamespace(
        surface_pressure=np.full((2, 3), 95_000.0),
        surface_qv=np.full((2, 3), 0.01),
    )
    met_fields = {
        "T2": np.full((2, 3), 290.0),
        "SKINTEMP": np.full((2, 3), 291.0),
        "U10": np.zeros((2, 4)),
        "V10": np.zeros((3, 3)),
        "LANDSEA": np.ones((2, 3)),
    }
    if field in {"surface_pressure", "surface_qv"}:
        setattr(result, field, value)
    else:
        met_fields[field] = value

    with pytest.raises(ValueError, match=error):
        _validate_prepared_near_surface(
            result, SimpleNamespace(fields=met_fields), cfg)


def test_near_surface_refusal_names_the_offending_cells_and_extremes():
    """The refusal has to say WHICH cells and HOW far, not just the bound.

    Field 2026-08-08: a nested HRRR tree over eastern Colorado refused
    with "prepared near-surface surface_qv is outside the physical range
    0.0..0.2" and nothing else, and reading the two offending cells back
    cost a full re-run of a two-stage preparation.  The numbers below are
    that run's real d02 minimum; with them in the message the excursion
    is legible as a two-cell WPS interpolation undershoot on sight,
    rather than as a possible unit error.
    """
    cfg = SimpleNamespace(ny=2, nx=3)
    surface_qv = np.full((2, 3), 0.0176926)
    surface_qv[0, 1] = -1.785645637e-05
    surface_qv[1, 2] = -1.478747851e-05
    result = SimpleNamespace(
        surface_pressure=np.full((2, 3), 95_000.0), surface_qv=surface_qv)
    met = SimpleNamespace(fields={
        "T2": np.full((2, 3), 290.0),
        "SKINTEMP": np.full((2, 3), 291.0),
        "U10": np.zeros((2, 4)),
        "V10": np.zeros((3, 3)),
        "LANDSEA": np.ones((2, 3)),
    })

    with pytest.raises(ValueError) as refusal:
        _validate_prepared_near_surface(result, met, cfg)

    message = str(refusal.value)
    assert "surface_qv is outside the physical range 0.0..0.2" in message
    assert "2 of 6 cell(s) are out" in message
    assert "-1.78565e-05" in message


def test_source_neutral_surface_contract_rejects_land_identity_drift():
    shape = (2, 3)
    fields = {
        name: np.ones((4, *shape) if name in {"TSLB", "SMOIS", "SH2O"}
                      else shape, dtype=np.float32)
        for name in runner._CANONICAL_SURFACE_FIELDS
    }
    fields["LANDMASK"][:] = 0.0
    fields["XLAND"][:] = 1.0

    with pytest.raises(ValueError, match="XLAND differs"):
        _validate_prepared_surface(
            SimpleNamespace(fields=fields),
            SimpleNamespace(ny=2, nx=3, sf_surface_physics=2,
                            num_soil_layers=4))


def test_output_due_thompson_consumes_native_refl_10cm_stash():
    state = SimpleNamespace(
        qv=object(), physics=SimpleNamespace(mp_physics=8))
    native = object()
    consumed = []

    result = runner._consume_due_native_refl_10cm(
        state, 1, lambda actual: consumed.append(actual) or native)

    assert result is native
    assert consumed == [state]
    assert runner._consume_due_native_refl_10cm(
        state, 0, lambda actual: pytest.fail("initial frame consumed a stash")) \
        is None


def _retime_single_domain(exp, *, cadence_seconds, run_seconds=None):
    run_seconds = (
        float(exp.run_seconds) if run_seconds is None else float(run_seconds))
    domain = replace(
        exp.root,
        history_interval_s=float(cadence_seconds),
        run=replace(
            exp.root.run,
            output_interval_s=float(cadence_seconds),
            run_seconds=run_seconds,
        ),
    )
    return replace(exp, run_seconds=run_seconds, domains=(domain,))


@pytest.mark.parametrize(
    ("cadence_seconds", "expected_offsets", "expected_suffixes",
     "last_equals_run_end"),
    (
        (900.0, [0.0, 900.0, 1800.0, 2700.0, 3600.0, 4500.0,
                 5400.0, 6300.0, 7200.0, 8100.0, 9000.0, 9900.0,
                 10800.0], ["00_00_00", "03_00_00"], True),
        (10800.0, [0.0, 10800.0], ["00_00_00", "03_00_00"], True),
        (7200.0, [0.0, 7200.0], ["00_00_00", "02_00_00"], False),
    ),
)
def test_hash_bound_history_cadence_uses_floor_schedule_for_any_due_output(
        cadence_seconds, expected_offsets, expected_suffixes,
        last_equals_run_end):
    base = load_experiment(ROOT / "configs" / "gfs_wrf_direct_proof.toml")
    exp = _retime_single_domain(base, cadence_seconds=cadence_seconds)

    receipt = runner._validate_hash_bound_history_cadence(
        exp, cadence_seconds)
    schedule = runner._history_output_schedule(
        start_time=exp.start_time, run_seconds=exp.run_seconds,
        cadence_seconds=cadence_seconds)

    assert [record[0] for record in schedule] == expected_offsets
    assert schedule[0][2].endswith(expected_suffixes[0])
    assert schedule[-1][2].endswith(expected_suffixes[1])
    assert receipt["expected_frame_count"] == len(schedule)
    assert receipt["last_scheduled_offset_seconds"] == expected_offsets[-1]
    assert receipt["last_scheduled_valid_time"] == schedule[-1][1].isoformat()
    assert receipt["run_end_offset_seconds"] == exp.run_seconds
    assert receipt["last_scheduled_equals_run_end"] is last_equals_run_end
    assert receipt["run_end_frame_scheduled"] is last_equals_run_end


def test_hash_bound_history_cadence_warns_on_cli_mismatch(capsys):
    """Warn-not-block: the hash-bound experiment cadence is what the
    run uses; a stale flag is named, not fatal.  The non-whole-step
    cadence stays a refusal (KEEP-HARD: the schedule cannot exist)."""
    base = load_experiment(ROOT / "configs" / "gfs_wrf_direct_proof.toml")
    subhourly = _retime_single_domain(base, cadence_seconds=900.0)
    receipt = runner._validate_hash_bound_history_cadence(subhourly, 1800.0)
    err = capsys.readouterr().err
    assert "warning:" in err and "authoritative" in err
    assert receipt["experiment_seconds"] == 900.0

    nonstep = _retime_single_domain(base, cadence_seconds=71.0)
    with pytest.raises(ValueError, match="whole number of exact model time steps"):
        runner._validate_hash_bound_history_cadence(nonstep, 71.0)


def test_cli_requires_explicit_source_physics_run_and_history_output_contract():
    args = runner._parse_args([
        "--source", "gfs",
        "--prepared-root", "prepared",
        "--proof-sha256", "1" * 64,
        "--source-manifest-sha256", "2" * 64,
        "--prepared-content-sha256", "3" * 64,
        "--experiment-config", "experiment.toml",
        "--wps-namelist", "namelist.wps",
        "--physics-profile", runner.PHYSICS_PROFILE,
        "--run-seconds", "10800",
        "--history-interval-seconds", "900",
        "--io-mode", "history",
        "--outdir", "run",
    ])
    assert args.source == "gfs"
    assert args.physics_profile == runner.PHYSICS_PROFILE
    assert args.io_mode == "history"
    assert args.history_interval_seconds == 900.0


def test_cli_accepts_explicit_nssl2_validation_candidate_profile():
    args = runner._parse_args([
        "--source", "era5",
        "--prepared-root", "prepared",
        "--proof-sha256", "1" * 64,
        "--source-manifest-sha256", "2" * 64,
        "--prepared-content-sha256", "3" * 64,
        "--experiment-config", "experiment.toml",
        "--wps-namelist", "namelist.wps",
        "--physics-profile", runner.NSSL2_PHYSICS_PROFILE,
        "--run-seconds", "43200",
        "--history-interval-seconds", "3600",
        "--io-mode", "history",
        "--outdir", "run",
    ])

    assert args.physics_profile == runner.NSSL2_PHYSICS_PROFILE


def test_cli_accepts_explicit_guarded_thompson_profile():
    args = runner._parse_args([
        "--source", "gfs",
        "--prepared-root", "prepared",
        "--proof-sha256", "1" * 64,
        "--source-manifest-sha256", "2" * 64,
        "--prepared-content-sha256", "3" * 64,
        "--experiment-config", "experiment.toml",
        "--wps-namelist", "namelist.wps",
        "--physics-profile", runner.THOMPSON_PHYSICS_PROFILE,
        "--run-seconds", "10800",
        "--history-interval-seconds", "3600",
        "--io-mode", "history",
        "--outdir", "run",
    ])

    assert args.physics_profile == runner.THOMPSON_PHYSICS_PROFILE


# ---------------------------------------------------------------------------
# First-time-user papercuts from the 2026-07-30 Linux pilots.
# ---------------------------------------------------------------------------

def _wizard_config(tmp_path):
    """A real wizard-emitted single-domain TOML (not a hand-built stub)."""
    from gpuwm.cli import main as cli_main

    out = tmp_path / "area.toml"
    rc = cli_main(["domain", "--point=25.76,-80.19", "--card", "24gb",
                   "--ladder", "12", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    assert rc == 0
    return out


def test_clock_flags_default_to_the_hash_bound_experiment(tmp_path, capsys):
    """PP-15: flags that can hold only one value must not be required.

    --run-seconds and --history-interval-seconds are validated to equal
    the hash-bound experiment exactly, so requiring the user to retype
    them bought nothing -- and cost a wasted cycle to anyone who tried
    to shorten a retry and was told the values must match.
    """
    config = _wizard_config(tmp_path)
    capsys.readouterr()

    run_seconds, cadence = runner._clock_defaults(config)
    exp = load_experiment(config)
    assert run_seconds == float(exp.run_seconds)
    assert cadence == float(exp.root.history_interval_s)

    args = SimpleNamespace(
        experiment_config=config, run_seconds=None,
        history_interval_seconds=None)
    runner._resolve_clock_arguments(args)
    assert args.run_seconds == run_seconds
    assert args.history_interval_seconds == cadence
    noted = capsys.readouterr().err
    assert "--run-seconds defaulted to" in noted
    assert "--history-interval-seconds defaulted to" in noted

    # Explicit values are left exactly alone (and stay subject to the
    # unchanged exact-match guard downstream).
    explicit = SimpleNamespace(
        experiment_config=config, run_seconds=1.0,
        history_interval_seconds=2.0)
    runner._resolve_clock_arguments(explicit)
    assert (explicit.run_seconds, explicit.history_interval_seconds) == (
        1.0, 2.0)
    assert capsys.readouterr().err == ""


def test_clock_flags_are_optional_in_the_parser(tmp_path):
    config = _wizard_config(tmp_path)
    args = runner._parse_args([
        "--source", "gfs", "--prepared-root", str(tmp_path / "prep"),
        "--proof-sha256", "0" * 64,
        "--source-manifest-sha256", "1" * 64,
        "--prepared-content-sha256", "2" * 64,
        "--experiment-config", str(config),
        "--wps-namelist", str(config.with_suffix(".namelist.wps")),
        "--physics-profile", next(iter(runner.PHYSICS_PROFILES)),
        "--io-mode", "history", "--outdir", str(tmp_path / "run"),
    ])
    assert args.run_seconds is None
    assert args.history_interval_seconds is None


def test_clock_defaults_refuse_a_config_they_cannot_read(tmp_path):
    bad = tmp_path / "not-an-experiment.toml"
    bad.write_text("[nothing]\nhere = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pass them explicitly"):
        runner._clock_defaults(bad)


def test_the_stability_abort_names_the_dominant_term_and_the_remedy():
    """The abort explains the co-located term and the dt remedy."""
    run = SimpleNamespace(dt=60.0, dx=12_000.0, ztop=20_000.0)
    state = SimpleNamespace()

    tropical = runner._stability_diagnosis(
        {"u_max": 29.3, "w_max": 6.62, "horizontal_cfl": 0.1465,
         "vertical_cfl": 12.4}, state, run)
    assert "VERTICAL Courant 12.40" in tropical
    assert "co-located" in tropical
    assert "REMEDY: retry with a lower dt" in tropical
    assert "time_step to about 15 s" in tropical
    assert "+22% wall time" in tropical

    # A genuine horizontal violation is named as such, not mislabelled.
    horizontal = runner._stability_diagnosis(
        {"u_max": 4000.0, "w_max": 0.1, "horizontal_cfl": 20.0,
         "vertical_cfl": 0.2}, state, run)
    assert "HORIZONTAL Courant 20.0" in horizontal
    assert "dx 12.000 km" in horizontal
    assert "REMEDY: retry with a lower dt" in horizontal


def test_the_materializer_accepts_the_wizard_s_own_advisory_fetch_table(
        tmp_path):
    """Node-3 #2: `unknown table(s)/top-level key(s) ['fetch']`.

    The wizard writes that advisory table into every config it emits,
    and `gpuwm check` and `rw-wps` both accept it -- only the
    materializer, which handed the raw dict to build_experiment, did
    not.  Materialization has to run BEFORE the front door, so this
    rejected the wizard's own output at the first step of the route.
    """
    config = _wizard_config(tmp_path)
    base = config.read_text(encoding="utf-8")
    assert "[fetch]" in base

    rendered, exp, receipt = runner._render_materialized_experiment(
        base, source="gfs",
        profile="morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1")
    # Preserved verbatim, not silently dropped: it is provenance.
    assert "[fetch]" in rendered
    assert receipt["profile_validation"]["profile"] == (
        "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1")
    assert (receipt["base_non_physics_descriptor_sha256"]
            == receipt["generated_non_physics_descriptor_sha256"])

    # A malformed [fetch] table is still refused rather than ignored.
    broken = base.replace('source = "gfs"', 'source = "not-a-source"')
    with pytest.raises(ValueError):
        runner._render_materialized_experiment(
            broken, source="gfs",
            profile="morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1")


# ---------------------------------------------------------------------------
# F3: the GFS source manifest this release's own fetch authors
# ---------------------------------------------------------------------------
#
# `1f0fc039 feat(fetch): the GFS ladder follows the case's model top` added
# `pressure_levels_hpa`/`top_pressure_pa` to the front-door manifest's `source`
# object.  This runner demanded that object be EXACTLY {model, product, cycle},
# so it refused every manifest v1.3.0's `gpuwm fetch` writes -- every GFS
# single-domain forecast on the release, not just one physics suite.
#
# The fixture rule applies: the manifest below is authored by the fetch lane's
# own writer over the captured NOMADS inventory in tests/fixtures/gfs-inventory,
# never hand-written here.  Nothing is downloaded.

_GFS_FETCH_CYCLE = datetime(2026, 7, 28, 6)
_GFS_FETCH_AREA = "30,-100,40,-90"


def _authored_gfs_front_door_manifest(
        tmp_path: Path, monkeypatch, *, top_pressure_pa: float | None = None,
) -> dict:
    """One real front-door manifest, written by `gpuwm fetch` itself."""

    import gpuwm.fetch as fetch
    from tools import download_gfs_native_subset as gfs_transport

    index_text = (ROOT / "tests" / "fixtures" / "gfs-inventory"
                  / "gfs.t12z.pgrb2.0p25.f000.idx").read_text(encoding="ascii")
    real_levels = gfs_transport.available_levels_from_index(index_text)

    def _grib2(messages: int) -> bytes:
        one = (b"GRIB" + b"\x00\x00" + b"\x00" + b"\x02"
               + (20).to_bytes(8, "big") + b"7777")
        return one * messages

    def download(url, destination, **_kw):
        from urllib.parse import parse_qsl, urlsplit
        query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
        count = len([key for key in query
                     if key.startswith("lev_") and key.endswith("_mb")])
        destination.write_bytes(
            _grib2(gfs_transport.record_count_for_levels(count)))

    monkeypatch.setattr(gfs_transport, "_download", download)
    out = tmp_path / "gfs"
    fetch.fetch_gfs(
        cycle=_GFS_FETCH_CYCLE, hours=(0, 3),
        area=fetch.parse_area(_GFS_FETCH_AREA), out=out,
        progress=lambda line: None,
        derived_bar=lambda cycle, **kw: fetch.gfs_derived_record_bar(
            cycle, index_text=index_text, levels_hpa=kw.get("levels_hpa"),
            progress=lambda line: None),
        available_levels=lambda cycle, **kw: real_levels,
        top_pressure_pa=top_pressure_pa)
    for name in ("bridge", "namelist.wps", "experiment.toml"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    path, _digest = fetch.author_gfs_front_door_manifest(
        out=out, bridge=tmp_path / "bridge",
        wps_namelist=tmp_path / "namelist.wps",
        experiment_config=tmp_path / "experiment.toml",
        progress=lambda line: None)
    return json.loads(path.read_text(encoding="utf-8"))


def _experiment_stub(*, start_time, p_top_pa: float):
    return SimpleNamespace(
        start_time=start_time, vertical=SimpleNamespace(p_top=p_top_pa))


#: The preparation proof these manifest-identity tests are read against.
#: A proof with no initial-condition provenance block declares no
#: forecast lead, which is exactly what every proof written before
#: lead-offset initialization meant: start_time IS the cycle.  The
#: manifest binding now derives the cycle as start_time minus the
#: declared lead, so the proof has to be named rather than assumed --
#: test_forecast_offset_init.py exercises the nonzero-lead half.
_ANALYSIS_START_PROOF: dict = {}


def test_the_runner_accepts_the_manifest_this_release_s_fetch_authors(
        tmp_path, monkeypatch):
    """F3: the producing lane's real output, through the consuming lane."""

    manifest = _authored_gfs_front_door_manifest(
        tmp_path, monkeypatch, top_pressure_pa=5000.0)

    # The producer really does write the level provenance -- this test is
    # worthless if the manifest happens not to carry the added keys.
    identity = manifest["source"]
    assert sorted(identity) == [
        "cycle", "model", "pressure_levels_hpa", "product", "top_pressure_pa"]
    assert len(identity["pressure_levels_hpa"]) == 23
    assert identity["top_pressure_pa"] == 5000.0

    roles, receipt = runner._manifest_file_specs(
        "gfs", manifest,
        _experiment_stub(start_time=_GFS_FETCH_CYCLE, p_top_pa=5000.0),
        _ANALYSIS_START_PROOF)

    assert {"series", "bridge", "wps_namelist", "experiment_config",
            "grib-f000", "grib-f003"} == set(roles)
    assert receipt["identity"] == {
        "model": "GFS", "product": "pgrb2.0p25",
        "cycle": "2026-07-28T06:00:00Z"}
    # Routed, not ignored: the ladder reaches the receipt and the derived
    # source top is the one the vertical contract is entitled to.
    assert receipt["pressure_levels_hpa"] == identity["pressure_levels_hpa"]
    assert receipt["source_top_pressure_pa"] == 5000.0
    assert receipt["experiment_p_top_pa"] == 5000.0


def test_the_runner_still_refuses_a_manifest_from_a_different_cycle(
        tmp_path, monkeypatch):
    """The control: identity stays three keys, compared strictly."""

    manifest = _authored_gfs_front_door_manifest(
        tmp_path, monkeypatch, top_pressure_pa=5000.0)
    other_cycle = _GFS_FETCH_CYCLE + timedelta(hours=6)

    with pytest.raises(ValueError, match="identity differs from the cycle this experiment"):
        runner._manifest_file_specs(
            "gfs", manifest,
            _experiment_stub(start_time=other_cycle, p_top_pa=5000.0),
            _ANALYSIS_START_PROOF)

    for key in ("model", "product"):
        wrong = json.loads(json.dumps(manifest))
        wrong["source"][key] = "something-else"
        with pytest.raises(ValueError,
                           match="identity differs from the cycle this experiment"):
            runner._manifest_file_specs(
                "gfs", wrong,
                _experiment_stub(start_time=_GFS_FETCH_CYCLE, p_top_pa=5000.0),
                _ANALYSIS_START_PROOF)


def test_unknown_manifest_identity_fields_warn_and_the_bound_set_decides(
        tmp_path, monkeypatch, capsys):
    """Warn-not-block: a NEWER fetch may stamp fields this runner does
    not read; they are named and ignored, and every field the runner
    DOES read is still bound and checked.  (The old closed-set refusal
    rejected manifests this release's own fetch authored.)"""

    manifest = _authored_gfs_front_door_manifest(
        tmp_path, monkeypatch, top_pressure_pa=5000.0)
    manifest["source"]["invented_by_nobody"] = 1

    specs, receipt = runner._manifest_file_specs(
        "gfs", manifest,
        _experiment_stub(start_time=_GFS_FETCH_CYCLE, p_top_pa=5000.0),
        _ANALYSIS_START_PROOF)
    err = capsys.readouterr().err
    assert "warning:" in err and "invented_by_nobody" in err
    assert receipt["identity"]["model"] == "GFS"


def test_the_runner_holds_the_level_ladder_to_the_vertical_contract(
        tmp_path, monkeypatch):
    """The level fields are validated where they matter, not just carried.

    A 50 hPa fetch cannot serve a 10 hPa model top, and the producing
    lane's own validator is what says so -- the same function
    `gpuwm/gfs_direct.py` calls before the bridge runs.
    """

    manifest = _authored_gfs_front_door_manifest(
        tmp_path, monkeypatch, top_pressure_pa=5000.0)

    with pytest.raises(ValueError, match="reaching 5000 Pa"):
        runner._manifest_file_specs(
            "gfs", manifest,
            _experiment_stub(start_time=_GFS_FETCH_CYCLE, p_top_pa=1000.0),
            _ANALYSIS_START_PROOF)

    broken = json.loads(json.dumps(manifest))
    broken["source"]["top_pressure_pa"] = 1234.0
    with pytest.raises(ValueError, match="pressure ladder tops out"):
        runner._manifest_file_specs(
            "gfs", broken,
            _experiment_stub(start_time=_GFS_FETCH_CYCLE, p_top_pa=1000.0),
            _ANALYSIS_START_PROOF)


def test_a_manifest_without_the_ladder_still_means_the_certified_top(
        tmp_path, monkeypatch):
    """Roots prepared before `1f0fc039` keep working, unchanged."""

    manifest = _authored_gfs_front_door_manifest(tmp_path, monkeypatch)
    for key in ("pressure_levels_hpa", "top_pressure_pa"):
        manifest["source"].pop(key)

    _roles, receipt = runner._manifest_file_specs(
        "gfs", manifest,
        _experiment_stub(start_time=_GFS_FETCH_CYCLE, p_top_pa=10000.0),
        _ANALYSIS_START_PROOF)

    assert receipt["pressure_levels_hpa"] is None
    assert receipt["source_top_pressure_pa"] == 10000.0


# ---------------------------------------------------------------------------
# The runner is part of the installed package, not a checkout script
# ---------------------------------------------------------------------------

def test_the_runner_is_importable_from_the_package():
    """`pip install gpuwm` has to be able to RUN a forecast.

    The chain's last two stages used to be a script path under
    ``tools/``, which only a git checkout has, so a wheel install could
    preprocess a GFS cycle and then had nowhere to run it.  "Install the
    product, then clone the repository to use it" is not an install.
    """

    import gpuwm.prepared_single_domain_forecast as packaged

    assert packaged.main is runner.main
    assert packaged.runner_capabilities()["runner"] \
        == "tools.prepared_single_domain_forecast"


def test_the_tools_entry_point_is_the_same_module_not_a_copy():
    """One implementation, two entries -- and receipts cannot diverge.

    ``tools/prepared_single_domain_forecast.py`` still works for every
    transcript and doc that names it.  It delegates: the module object
    it exposes IS the package module, so there is no second runner whose
    hashes could disagree with the first one's.
    """

    import gpuwm.prepared_single_domain_forecast as packaged

    assert runner is packaged


def test_the_runtime_identity_binds_the_packaged_runner_file():
    """The receipt names where the code actually lives."""

    identity = runner._runtime_source_identity()
    keys = set(identity["source_sha256"])
    assert "gpuwm/prepared_single_domain_forecast.py" in keys
    assert "tools/prepared_single_domain_forecast.py" not in keys


def test_a_reused_authority_directory_is_a_sentence_not_a_traceback(
        tmp_path, capsys):
    """Re-running the documented command is the commonest way here.

    ``--output-directory`` is create-only on purpose: an authority
    directory that merged two runs would publish a receipt describing
    neither.  But the refusal arrived as a raw ``FileExistsError``
    traceback ending in a Windows error number, which is how an owner
    met it.  One sentence, the remedy, exit 2 -- the same standard as
    the front-door refusals.
    """

    argv = [
        "--materialize-authorities", "--source", "gfs",
        "--base-experiment-config",
        str(_no_physics_copy(
            tmp_path, ROOT / "configs" / "gfs_wrf_direct_proof.toml")),
        "--base-wps-namelist",
        str(ROOT / "configs" / "gfs_wrf_direct_proof.namelist.wps"),
        "--physics-profile", runner.MORRISON_PHYSICS_PROFILE,
        "--output-directory", str(tmp_path / "authority")]
    assert runner.main(argv) == 0
    capsys.readouterr()

    rc = runner.main(argv)
    captured = capsys.readouterr()
    assert rc == 2
    assert "Traceback" not in captured.err
    assert captured.err.strip().count("\n") == 0, captured.err
    assert "--output-directory refused" in captured.err
    # "already HOLDS", not "already exists": the first run populated it,
    # and holding an earlier run is the breakage.  A directory that
    # merely exists is accepted -- `gpuwm sim` hands this runner one it
    # claimed itself, every time.
    assert "already holds" in captured.err
    assert "Pass a new --output-directory" in captured.err


def test_the_physics_drift_refusal_reaches_the_command_line_layered(
        tmp_path, capsys):
    """rc 2, no traceback, and the explain marker never on the terminal.

    This stage's error path printed ``str(error)`` raw, which was fine
    while nothing it raised was layered.  The drift refusal is, so the
    path renders now: the action half plus a pointer by default, both
    halves under ``--explain``, and ``[[explain]]`` in neither.
    """

    argv = [
        "--materialize-authorities", "--source", "gfs",
        "--base-experiment-config",
        str(ROOT / "configs" / "gfs_wrf_direct_proof.toml"),
        "--base-wps-namelist",
        str(ROOT / "configs" / "gfs_wrf_direct_proof.namelist.wps"),
        "--physics-profile", runner.MORRISON_PHYSICS_PROFILE,
        "--output-directory", str(tmp_path / "authority")]

    assert runner.main(argv) == 2
    terse = capsys.readouterr().err
    assert "Traceback" not in terse
    assert "[[explain]]" not in terse
    assert "[shared] mp_physics: config 6, profile 10" in terse
    assert "--explain for the reason" in terse
    assert "binds by hash" not in terse

    assert runner.main(argv + ["--explain"]) == 2
    full = capsys.readouterr().err
    assert "[[explain]]" not in full
    assert "[shared] mp_physics: config 6, profile 10" in full
    assert "binds by hash" in full
    assert not (tmp_path / "authority").exists()


def test_a_missing_base_input_is_a_sentence_too(tmp_path, capsys):
    """The other way to mistype this stage's flags."""

    rc = runner.main([
        "--materialize-authorities", "--source", "gfs",
        "--base-experiment-config", str(tmp_path / "nope.toml"),
        "--base-wps-namelist", str(tmp_path / "nope.namelist.wps"),
        "--physics-profile", runner.MORRISON_PHYSICS_PROFILE,
        "--output-directory", str(tmp_path / "authority")])
    captured = capsys.readouterr()
    assert rc == 2
    assert "Traceback" not in captured.err
    assert "nope.toml" in captured.err


# ---------------------------------------------------------------------------
# The mp8 table gap, named at step 2 of 6 instead of after preprocessing
# ---------------------------------------------------------------------------

def test_materializing_mp8_without_tables_refuses_before_any_fetch(
        tmp_path, monkeypatch, capsys):
    """`FileNotFoundError: missing Thompson table asset ...qr_acr_qg_V4.dat`

    A wheel user met that traceback at the top of the GPU run, having
    already paid for the fetch and for rw-wps.  Everything needed to
    know it was coming was available at ``--materialize-authorities``,
    which is step 2 of the six the documentation prints and runs in
    under a second.  It is answered there now, in one sentence.
    """

    from gpuwm.table_assets import MissingTableAssets

    empty = tmp_path / "no-tables"
    empty.mkdir()
    monkeypatch.setenv(runner.THOMPSON_TABLE_ROOT_ENV, str(empty))
    base_experiment = _no_physics_copy(
        tmp_path, ROOT / "configs" / "gfs_wrf_direct_proof.toml")
    base_wps = ROOT / "configs" / "gfs_wrf_direct_proof.namelist.wps"

    with pytest.raises(MissingTableAssets) as raised:
        runner.materialize_named_source_authorities(
            source="gfs",
            base_experiment_config=base_experiment,
            base_wps_namelist=base_wps,
            physics_profile=runner.THOMPSON_PHYSICS_PROFILE,
            output_directory=tmp_path / "authority")
    message = str(raised.value)
    assert "mp_physics=8" in message
    assert "gpuwm fetch-tables" in message
    # Refused BEFORE anything was published: no half-written authority
    # directory for the next command to bind.
    assert not (tmp_path / "authority").exists()

    # Through the CLI it is a sentence and exit 2, never a traceback.
    assert runner.main([
        "--materialize-authorities", "--source", "gfs",
        "--base-experiment-config", str(base_experiment),
        "--base-wps-namelist", str(base_wps),
        "--physics-profile", runner.THOMPSON_PHYSICS_PROFILE,
        "--output-directory", str(tmp_path / "authority-cli"),
    ]) == 2
    printed = capsys.readouterr().err
    assert "Traceback" not in printed
    assert "prepared_single_domain_forecast: refused:" in printed
    assert "gpuwm fetch-tables" in printed


def test_a_non_mp8_suite_never_asks_for_thompson_tables(tmp_path,
                                                        monkeypatch):
    """Negative control: the gate is keyed on the selector, not the run.

    With no tables anywhere, a Morrison authority still materializes --
    otherwise the refusal above would be a blanket block on every
    forecast rather than the one scheme that reads those files.
    """

    empty = tmp_path / "no-tables"
    empty.mkdir()
    monkeypatch.setenv(runner.THOMPSON_TABLE_ROOT_ENV, str(empty))
    receipt = runner.materialize_named_source_authorities(
        source="gfs",
        base_experiment_config=_no_physics_copy(
            tmp_path, ROOT / "configs" / "gfs_wrf_direct_proof.toml"),
        base_wps_namelist=(
            ROOT / "configs" / "gfs_wrf_direct_proof.namelist.wps"),
        physics_profile=runner.MORRISON_PHYSICS_PROFILE,
        output_directory=tmp_path / "authority")
    assert receipt["status"] == "PASS"


# ---------------------------------------------------------------------------
# The prepared front door must accept the cache it just wrote (#198).
#
# ``_validate_cache_metadata`` compares ``user`` to ``expected_user``
# EXACTLY.  Any key a preparer records that the expectation omits is
# therefore a refusal, not a tolerated extra -- and the ERA5 preparer
# records ``soil_moisture_floor`` whenever the SMCDRY floor fires, which
# is the common case on clipped ERA5.
# ---------------------------------------------------------------------------

_FLOOR_RECEIPT = {"floored_land_cells": 12, "min_pre_floor": 0.0021}
_REPAIR_RECEIPT = {"repaired_land_cells": 3, "non_finite_cells": 0}


def _cache_metadata_case(*, user_extra=None, proof_extra=None):
    """A portable-single-domain cache/proof pair that validates."""

    start = datetime(2024, 5, 20, 18, 0)
    forcing_hours = [0, 1]
    metadata = {
        "user": {
            "source_adapter": "era5",
            "initial_valid_time": start.isoformat(),
            "last_valid_time": (start + timedelta(hours=1)).isoformat(),
            "forcing_hours": list(forcing_hours),
            "boundary_interval_seconds": 3600,
            "preprocessing": {"backend": "cpu"},
            **(user_extra or {}),
        },
        "surface_fields": sorted(runner._CANONICAL_SURFACE_FIELDS),
        "met_fields": sorted(runner._REQUIRED_MET_FIELDS),
        "lbc": {
            "spec_bdy_width": 5,
            "spec_zone": 1,
            "relax_zone": 4,
            "intervals": [{
                "start_seconds": 0.0,
                "end_seconds": 3600.0,
                "fields": list(runner._LBC_FIELDS),
            }],
        },
    }
    proof = {
        "preprocessing": {"backend": "cpu"},
        **(proof_extra or {}),
    }
    exp = SimpleNamespace(
        start_time=start,
        root=SimpleNamespace(run=SimpleNamespace(
            spec_bdy_width=5, spec_zone=1, relax_zone=4)))
    return dict(
        reader=SimpleNamespace(header={"metadata": metadata}),
        source="era5", exp=exp, forcing_hours=forcing_hours,
        boundary_interval_seconds=3600, proof=proof,
        layout="portable-single-domain-v2")


def test_a_cache_written_with_the_smcdry_floor_fired_validates():
    """#198: the floor firing must not make the front door refuse itself.

    Before the fix this raised ``prepared cache user metadata differs
    from source/proof/experiment`` -- the preparation wrote the receipt,
    the expectation did not list it, and the exact comparison refused.
    """

    runner._validate_cache_metadata(**_cache_metadata_case(
        user_extra={"soil_moisture_floor": _FLOOR_RECEIPT},
        proof_extra={"soil_moisture_floor": _FLOOR_RECEIPT}))


def test_the_deep_soil_repair_receipt_validates_the_same_way():
    """The second instance of the same class, fixed by the same tuple."""

    runner._validate_cache_metadata(**_cache_metadata_case(
        user_extra={"soil_moisture_floor": _FLOOR_RECEIPT,
                    "deep_soil_repair": _REPAIR_RECEIPT},
        proof_extra={"soil_moisture_floor": _FLOOR_RECEIPT,
                     "deep_soil_repair": _REPAIR_RECEIPT}))


def test_a_floor_receipt_that_differs_from_its_proof_still_refuses():
    """Strictness kept: the fix binds to the proof, it does not wave through."""

    with pytest.raises(ValueError, match="user metadata differs"):
        runner._validate_cache_metadata(**_cache_metadata_case(
            user_extra={"soil_moisture_floor": _FLOOR_RECEIPT},
            proof_extra={"soil_moisture_floor": {"floored_land_cells": 99}}))


def test_a_cache_receipt_absent_from_the_proof_still_refuses():
    """A cache claiming a floor its proof never recorded is inconsistent."""

    with pytest.raises(ValueError, match="user metadata differs"):
        runner._validate_cache_metadata(**_cache_metadata_case(
            user_extra={"soil_moisture_floor": _FLOOR_RECEIPT}))


def test_a_proof_receipt_absent_from_the_cache_still_refuses():
    """And the mirror: the proof recorded a floor the cache did not."""

    with pytest.raises(ValueError, match="user metadata differs"):
        runner._validate_cache_metadata(**_cache_metadata_case(
            proof_extra={"soil_moisture_floor": _FLOOR_RECEIPT}))


def test_a_genuinely_mismatched_knob_is_still_refused():
    """Negative control: the fix did not loosen the ordinary comparison."""

    case = _cache_metadata_case()
    case["boundary_interval_seconds"] = 1800
    with pytest.raises(ValueError, match="user metadata differs"):
        runner._validate_cache_metadata(**case)


def test_a_cache_with_no_receipts_is_unchanged_by_the_fix():
    """The silent path stays silent: no key, no expectation, no refusal."""

    runner._validate_cache_metadata(**_cache_metadata_case())


def test_the_era5_writer_and_the_validator_share_one_receipt_list():
    """The CLASS fix: one tuple, imported by both sides.

    This is the test that would have caught #198 at authoring time --
    the writer's key set and the validator's expectation are the same
    object, so they cannot drift apart again.
    """

    from gpuwm import era5_direct
    from gpuwm.ingest.prepared_cache import CONDITIONAL_PREPARATION_RECEIPTS

    assert "soil_moisture_floor" in CONDITIONAL_PREPARATION_RECEIPTS
    assert "deep_soil_repair" in CONDITIONAL_PREPARATION_RECEIPTS
    # Every metadata key the writer can bind resolves to an attribute it
    # actually reads off the soil result.
    for key in CONDITIONAL_PREPARATION_RECEIPTS:
        attribute = era5_direct._SOIL_RECEIPT_ATTRIBUTE.get(key, key)
        assert isinstance(attribute, str) and attribute


def test_the_shared_mapped_evidence_validator_is_not_named_for_one_source():
    """One validator serves EVERY packaged mapped profile, so its name
    may not claim one of them.

    It was authored as ``_validate_twentycrv3_mapped_evidence`` when
    20CRv3 was the only packaged mapped profile.  It now certifies GDAS,
    RRFS, AIGFS, AIGEFS, GEFS and the hybrids too -- a 20CRv3-named
    function in an AIGEFS traceback is how a reader concludes the wrong
    route ran.  The name is generic; this test keeps it generic as more
    sources join ``_MAPPED_PACKAGED_PROFILE``.
    """

    from gpuwm import prepared_single_domain_forecast as module

    assert not hasattr(module, "_validate_twentycrv3_mapped_evidence")
    name = module._validate_packaged_mapped_evidence.__name__.lower()
    tokens = {"twentycrv3"}
    for source in module._MAPPED_PACKAGED_PROFILE:
        tokens.add(source.replace("-", "").replace("_", "").lower())
    assert tokens, "no packaged mapped profile is declared"
    for token in tokens:
        assert token not in name, (
            f"the shared mapped-evidence validator is named for {token}")


def _raised_message_strings(function) -> list[str]:
    """Every string literal that reaches a ``raise`` in ``function``.

    Docstrings and comments are deliberately excluded: prose may name the
    source a behaviour was first measured on.  A REFUSAL may not, because
    the refusal is what the user reads.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    messages: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Constant) and isinstance(inner.value,
                                                              str):
                messages.append(inner.value)
    return messages


def test_the_shared_mapped_refusals_name_no_single_source():
    """A refusal that hardcodes one source lies about every other one.

    The shared validator certifies every packaged mapped profile, but its
    refusals were written when 20CRv3 was the only one -- so a GEFS or
    AIGEFS bundle failing its soil-layer receipt was refused with a
    sentence naming 20CRv3.  A reader debugging an AIGEFS run reasonably
    concludes the wrong route ran, which is the same defect the validator
    RENAME closed one level up.

    The refusal CLASSES and their remedies are untouched; only the source
    the sentence names is now the source that actually failed.
    """

    from gpuwm import prepared_single_domain_forecast as module

    tokens = {"20crv3", "twentycrv3"}
    for source in module._MAPPED_PACKAGED_PROFILE:
        tokens.add(source.replace("-", "").replace("_", "").lower())

    offenders = []
    for function in (module._validate_packaged_mapped_evidence,
                     module._validate_mapped_static_proof):
        for message in _raised_message_strings(function):
            flattened = message.replace("-", "").replace("_", "").lower()
            for token in tokens:
                if token in flattened:
                    offenders.append((function.__name__, message, token))
    assert not offenders, (
        "these refusals name one source inside a shared validator: "
        f"{offenders}")
