"""Fetch acquisition closes the default-physics input dependency."""
import argparse
import bz2
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import time

import pytest


def _empty_cache(tmp_path, monkeypatch):
    for key in ("GPUWM_WIF_CLIMATOLOGY", "GPUWM_WIF_CLIMATOLOGY_ROOT"):
        monkeypatch.delenv(key, raising=False)
    cache = tmp_path / "cache"
    monkeypatch.setenv("GPUWM_WIF_DATA_ROOT", str(cache))
    return cache


def _experiment(tmp_path):
    from test_runplan_hrrr_tree import _plan
    from gpuwm.runplan import resolve_plan

    return resolve_plan(_plan(tmp_path), require_inputs=False)


def test_default_plan_declares_the_asset_without_acquiring_it(tmp_path, monkeypatch):
    from gpuwm import table_assets

    cache = _empty_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(table_assets, "stage_wif_dataset", lambda: pytest.fail("planning downloaded data"))
    document, experiment, _ = _experiment(tmp_path)
    pending = next(row for row in document["automatic_resolutions"]
                   if row["key"] == "wif_climatology")
    assert pending["bytes"] == 225443520
    assert pending["sha256"] == "2f828eabd96a45f3872390f901240ea2259a1e9a629247010f42ce7a31cc46be"
    assert not cache.exists()
    assert experiment.root.run.mp_physics == 28
    assert experiment.root.run.bl_mynn_version == "gsd_41"
    assert experiment.root.run.mp28_aerosol_source != "synthetic"


def test_the_chain_passes_the_acquisition_flag_to_its_fetch(tmp_path, monkeypatch):
    from test_runplan_hrrr_tree import _drive, _stage

    _empty_cache(tmp_path, monkeypatch)
    stages, _, _, _ = _drive(tmp_path, monkeypatch)
    assert "--wif" in _stage(stages, "fetch")


def _staged_case(tmp_path, source):
    from gpuwm.domain_wizard import render_config
    from gpuwm.runplan import PLAN_SCHEMA, build_plan

    cycle = "1999-05-03T12" if source == "20crv3" else "2024-05-03T12"
    config = tmp_path / "case.toml"
    config.write_text(render_config(
        name="staged-wif", start_time=datetime.fromisoformat(cycle), hours=6,
        projection={"map_proj": "lambert", "ref_lat": 38.5, "ref_lon": -97.5,
                    "stand_lon": -97.5, "truelat1": 38.5, "truelat2": 38.5},
        dims=[(50, 50)], ratios=(), root_dx_m=3000.0,
        profile="thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1",
        fetch_hints={"source": source, "cycle": cycle, "hours": 6}, case_data=None),
        encoding="utf-8")
    # Dispatch stops at its captured fetch before reading these preparation inputs.
    config.with_suffix(".namelist.wps").write_text("&share /\n", encoding="utf-8")
    document = {"schema": PLAN_SCHEMA, "name": "staged-wif", "route": "prepared",
                "config": {"path": str(config)}, "output_root": str(tmp_path / "run"),
                "run_options": {"geog_root": str(tmp_path / "GEOG")}}
    plan = build_plan(document, source="synthetic staged WIF acquisition gate", base_dir=tmp_path,
                      sha256=hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest())
    return plan, config


def _forbid_acquisition(monkeypatch):
    from gpuwm import table_assets

    def unexpected(*args, **kwargs):
        pytest.fail("planning or dispatch acquired an actual asset")
    monkeypatch.setattr(table_assets, "stage_wif_dataset", unexpected)
    monkeypatch.setattr(table_assets, "_transfer", unexpected)


@pytest.mark.parametrize("source", ("hrrr-prs", "hrrr-native", "rap-native"))
def test_mapped_plan_declares_the_asset_without_acquiring_it(tmp_path, monkeypatch, source):
    from gpuwm.runplan import resolve_plan

    cache = _empty_cache(tmp_path, monkeypatch)
    _forbid_acquisition(monkeypatch)
    plan, _ = _staged_case(tmp_path, source)
    document, experiment, _ = resolve_plan(plan, require_inputs=False)
    pending = next(row for row in document["automatic_resolutions"]
                   if row["key"] == "wif_climatology")
    assert pending["value"] == str(cache / "QNWFA_QNIFA_SIGMA_MONTHLY.dat")
    assert pending["domains"] == [experiment.root.grid_id]
    assert experiment.root.run.mp_physics == 28 and experiment.root.run.specified
    assert not cache.exists()


@pytest.mark.parametrize("source", ("hrrr-prs", "hrrr-native", "rap-native"))
def test_mapped_dispatch_passes_the_acquisition_flag_before_preparation(tmp_path, monkeypatch, source):
    from test_runplan_hrrr_tree import _Observer
    from gpuwm import runplan

    cache = _empty_cache(tmp_path, monkeypatch)
    _forbid_acquisition(monkeypatch)
    plan, config = _staged_case(tmp_path, source)
    plan = replace(plan, run_options={**plan.run_options, "as_posted": False})
    experiment = runplan.resolve_plan(plan, require_inputs=False)[1]
    captured = []
    class CapturedFetch(Exception):
        pass
    def capture(arguments, run_dir, **kwargs):
        captured.extend(arguments)
        raise CapturedFetch()
    monkeypatch.setattr(runplan, "_run_fetch", capture)
    with pytest.raises(CapturedFetch):
        runplan._staged_chain(plan, config_path=config, exp=experiment,
                             observer=_Observer(), run_dir=plan.run_dir)
    assert captured[captured.index("--source") + 1] == source
    assert "--wif" in captured and "--whole-cycle" in captured
    assert not cache.exists()


@pytest.mark.parametrize("choice", ("config", "file_env", "root_env"))
def test_mapped_plan_keeps_a_missing_named_dataset_refusal(tmp_path, monkeypatch, choice):
    from gpuwm.runplan import resolve_plan

    _empty_cache(tmp_path, monkeypatch)
    _forbid_acquisition(monkeypatch)
    plan, config = _staged_case(tmp_path, "hrrr-native")
    missing = tmp_path / "chosen.dat"
    if choice == "config":
        config.write_text(config.read_text(encoding="utf-8").replace(
            "[shared]\n", "[shared]\nwif_climatology_path = " + json.dumps(str(missing)) + "\n"),
            encoding="utf-8")
    else:
        key = "GPUWM_WIF_CLIMATOLOGY" if choice == "file_env" else "GPUWM_WIF_CLIMATOLOGY_ROOT"
        monkeypatch.setenv(key, str(missing))
    with pytest.raises(ValueError, match="chosen.dat"):
        resolve_plan(plan, require_inputs=False)


def test_mapped_offline_opt_out_keeps_the_missing_dataset_refusal(tmp_path, monkeypatch):
    from gpuwm.runplan import resolve_plan

    _empty_cache(tmp_path, monkeypatch)
    _forbid_acquisition(monkeypatch)
    plan, config = _staged_case(tmp_path, "hrrr-native")
    config.write_text(config.read_text(encoding="utf-8").replace(
        "[fetch]\n", "[fetch]\nwif = false\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="fetch-tables --wif"):
        resolve_plan(plan, require_inputs=False)


def test_mapped_local_inputs_cannot_defer_the_missing_dataset(tmp_path, monkeypatch):
    from gpuwm import local_preparation
    from gpuwm.runplan import resolve_plan

    _empty_cache(tmp_path, monkeypatch)
    _forbid_acquisition(monkeypatch)
    plan, config = _staged_case(tmp_path, "20crv3")
    root = tmp_path / "local-inputs"
    config.write_text(config.read_text(encoding="utf-8").replace(
        "[fetch]\n", "[fetch]\nsource_root = " + json.dumps(str(root)) + "\n"), encoding="utf-8")
    monkeypatch.setattr(local_preparation, "review_local_inputs", lambda *args, **kwargs: {
        "source_root": str(root), "sha256": "0" * 64, "files": [], "file_count": 0})
    with pytest.raises(ValueError, match="fetch-tables --wif"):
        resolve_plan(plan, require_inputs=False)


def test_named_dataset_is_not_replaced_by_an_automatic_download(tmp_path, monkeypatch):
    from gpuwm.config import validate_experiment_preparation
    from gpuwm.preparation_assets import wif_fetch_domains

    _empty_cache(tmp_path, monkeypatch)
    _, experiment, _ = _experiment(tmp_path)
    missing = tmp_path / "chosen.dat"
    experiment = replace(experiment, domains=tuple(
        replace(domain, run=replace(domain.run, wif_climatology_path=str(missing)))
        for domain in experiment.domains))
    assert wif_fetch_domains(experiment, {"source": "hrrr"}) == ()
    with pytest.raises(ValueError, match="chosen.dat"):
        validate_experiment_preparation(experiment)


def test_a_pending_dataset_does_not_suppress_other_preconditions(tmp_path, monkeypatch):
    from gpuwm import config
    from gpuwm.preparation_assets import wif_fetch_domains

    _empty_cache(tmp_path, monkeypatch)
    _, experiment, _ = _experiment(tmp_path)
    pending = wif_fetch_domains(experiment, {"source": "hrrr"})
    monkeypatch.setattr(config, "RUN_PREPARATION_PRECONDITIONS",
                        config.RUN_PREPARATION_PRECONDITIONS + (lambda cfg: "another input is absent",))
    with pytest.raises(ValueError, match="another input"):
        config.validate_experiment_preparation(experiment, pending_wif_domains=pending)


def test_an_offline_opt_out_keeps_the_actionable_refusal(tmp_path, monkeypatch):
    from gpuwm.config import validate_experiment_preparation
    from gpuwm.preparation_assets import wif_fetch_domains

    _empty_cache(tmp_path, monkeypatch)
    _, experiment, _ = _experiment(tmp_path)
    assert wif_fetch_domains(experiment, {"source": "hrrr", "wif": False}) == ()
    with pytest.raises(ValueError, match="fetch-tables --wif"):
        validate_experiment_preparation(experiment)


def test_fetch_stops_before_forcing_when_the_asset_cannot_be_acquired(tmp_path, monkeypatch, capsys):
    from gpuwm import fetch, table_assets

    parser = argparse.ArgumentParser()
    fetch.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["fetch", "--source", "hrrr", "--cycle", "2024-05-03T12",
                              "--hours", "1", "--out", str(tmp_path), "--wif"])
    monkeypatch.setattr(table_assets, "stage_wif_dataset", lambda: 2)
    monkeypatch.setattr(fetch, "fetch_hrrr", lambda **kwargs: pytest.fail("forcing started after asset failure"))
    assert fetch.fetch_main(args) == 2
    message = capsys.readouterr().err
    assert "zero-inflow" in message and "fetch-tables --wif --from DIR" in message


def test_a_shared_asset_cache_downloads_only_once(tmp_path, monkeypatch):
    from gpuwm import table_assets
    from gpuwm.core.thompson_contract import TableAsset

    payload = b"pinned input transaction"
    asset = TableAsset("asset.dat", len(payload), hashlib.sha256(payload).hexdigest())
    calls = []
    def transfer(url, temp, pinned):
        calls.append(url)
        time.sleep(0.05)
        temp.write_bytes(payload)
    monkeypatch.setattr(table_assets, "_transfer", transfer)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: table_assets.fetch_asset_from_url(tmp_path, asset, "https://example.invalid/asset"), range(2)))
    assert results == [tmp_path / asset.filename] * 2
    assert calls == ["https://example.invalid/asset"]


def test_a_corrupt_cached_asset_is_refused_and_preserved(tmp_path, monkeypatch):
    from gpuwm import table_assets
    from gpuwm.core.thompson_contract import TableAsset

    payload = b"correct"
    asset = TableAsset("asset.dat", len(payload), hashlib.sha256(payload).hexdigest())
    file = tmp_path / asset.filename
    file.write_bytes(b"changed")
    monkeypatch.setattr(table_assets, "_transfer", lambda *args: pytest.fail("corrupt cache overwritten"))
    with pytest.raises(table_assets.TableAssetError, match="SHA-256"):
        table_assets.fetch_asset_from_url(tmp_path, asset, "https://example.invalid/asset")
    assert file.read_bytes() == b"changed"


def test_compressed_acquisition_verifies_the_raw_bytes_before_install(tmp_path):
    from gpuwm import table_assets
    from gpuwm.core.thompson_contract import TableAsset

    payload = b"monthly input transport" * 30
    source = tmp_path / "published.bz2"
    source.write_bytes(bz2.compress(payload))
    cache = tmp_path / "cache"
    cache.mkdir()
    asset = TableAsset("monthly.dat", len(payload), hashlib.sha256(payload).hexdigest())
    landed = table_assets.fetch_asset_from_url(cache, asset, source.as_uri(), compression="bz2")
    assert landed.read_bytes() == payload
    assert list(cache.glob("*fetch-partial*")) == []


def test_a_bad_compressed_download_leaves_no_installed_or_partial_file(tmp_path):
    from gpuwm import table_assets
    from gpuwm.core.thompson_contract import TableAsset

    source = tmp_path / "published.bz2"
    source.write_bytes(bz2.compress(b"incorrect bytes"))
    cache = tmp_path / "cache"
    cache.mkdir()
    asset = TableAsset("monthly.dat", 15, "0" * 64)
    with pytest.raises(table_assets.TableAssetError, match="SHA-256"):
        table_assets.fetch_asset_from_url(cache, asset, source.as_uri(), compression="bz2")
    assert not (cache / asset.filename).exists()
    assert list(cache.glob("*fetch-partial*")) == []
