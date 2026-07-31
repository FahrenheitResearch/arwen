"""Public HRRR wrapper source-window orchestration regressions.

The cold-start receipt this wrapper validates is **not** hand-built here.
It used to be, and that is exactly how a one-line schema desync shipped
through 800+ green tests: ``tools/hrrr_single_domain_benchmark.py`` bumped
the receipt to ``...-initialization-v2`` while this consumer still demanded
``v1``, so the documented HRRR preparation front door raised ``RuntimeError:
HRRR preparation omitted deterministic cold-start evidence`` *after* a fully
successful preparation, for every one of the seven profiles -- and the suite
could not see it, because the fixture below constructed the receipt it
wanted rather than the one the producer emits.

So the fixture now RUNS the producer: real decoded native fields through
``gpuwm.ingest.real.initialize_real`` and then through
``_initial_hrrr_microphysics_receipt``, the same function the benchmark
binds into ``report.json``.  A future schema bump, a renamed field, or a
dropped correspondence breaks this consumer's tests the moment the producer
changes, which is the only property that would have caught the regression.
"""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path

import pytest

from gpuwm.physics_compat import (
    KESSLER_PROFILE_ID,
    MORRISON_PROFILE_ID,
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    NSSL2_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    WSM6_PROFILE_ID,
)
from tools import prepare_hrrr_wrf as prepare
from tools.hrrr_single_domain_benchmark import (
    _initial_hrrr_microphysics_receipt)

# The real decoded-source -> initialized-state driver.  It lives beside the
# producer's own tests; importing it is what makes this a cross-lane seam
# test instead of a second hand-built fixture.
from test_hrrr_single_domain_benchmark import (
    _decoded_native_hrrr_initialization)


#: The microphysics option each fixed profile pins, so the producer runs the
#: scheme whose cold start the consumer is about to validate.
_PROFILE_MICROPHYSICS = {
    KESSLER_PROFILE_ID: 1,
    WSM6_PROFILE_ID: 6,
    THOMPSON_PROFILE_ID: 8,
    MORRISON_PROFILE_ID: 10,
    NSSL2_PROFILE_ID: 18,
    NSSL2_LEGACY_RRTMG_PROFILE_ID: 18,
}


@functools.lru_cache(maxsize=None)
def _producer_receipt(profile: str) -> dict[str, object]:
    """Whatever the real producer emits for ``profile``, verbatim."""

    result = _decoded_native_hrrr_initialization(
        _PROFILE_MICROPHYSICS[profile])
    return _initial_hrrr_microphysics_receipt(
        result.state, profile, result.hydrometeor_initialization)


def _cold_start_receipt(profile: str) -> dict[str, object]:
    return copy.deepcopy(_producer_receipt(profile))


@pytest.mark.parametrize("profile", sorted(_PROFILE_MICROPHYSICS))
def test_the_consumer_accepts_exactly_what_the_producer_emits(profile: str):
    """The binding the schema desync escaped through.

    Nothing here is written down twice: the receipt comes from the producer
    and the constant comes from the consumer, so they cannot drift apart
    silently again.
    """

    receipt = _producer_receipt(profile)
    assert receipt["schema"] == prepare.HRRR_INITIALIZATION_SCHEMA
    assert prepare._validated_physics_receipt(
        {"physics": {
            "schema": "gpuwm-prepared-physics-profile-v1",
            "profile": profile,
            "hrrr_initialization": receipt,
        }},
        requested_profile=profile)["profile"] == profile


def test_the_consumer_refuses_the_previous_producer_schema():
    """v1/v2 acceptance is deliberately not retained.

    ``main`` runs the producer itself and reads back the ``report.json``
    that invocation just wrote, so an older-schema receipt cannot reach this
    consumer; accepting one would widen the gate for a case that does not
    exist.  Dropping the version is therefore a refusal, not a widening.
    """

    for stale in ("gpuwm-hrrr-microphysics-initialization-v1",
                  "gpuwm-hrrr-microphysics-initialization-v2"):
        receipt = _cold_start_receipt(NSSL2_PROFILE_ID)
        receipt["schema"] = stale
        with pytest.raises(RuntimeError, match="cold-start evidence"):
            prepare._validated_physics_receipt(
                {"physics": {
                    "schema": "gpuwm-prepared-physics-profile-v1",
                    "profile": NSSL2_PROFILE_ID,
                    "hrrr_initialization": receipt,
                }},
                requested_profile=NSSL2_PROFILE_ID)


def test_the_consumer_validates_the_correspondence_fields_not_the_string():
    """Every v2/v3 field the producer added is checked, one at a time."""

    def _refuse(mutate, match="correspondence evidence"):
        receipt = _cold_start_receipt(NSSL2_PROFILE_ID)
        mutate(receipt)
        with pytest.raises(RuntimeError, match=match):
            prepare._validated_physics_receipt(
                {"physics": {
                    "schema": "gpuwm-prepared-physics-profile-v1",
                    "profile": NSSL2_PROFILE_ID,
                    "hrrr_initialization": receipt,
                }},
                requested_profile=NSSL2_PROFILE_ID)

    _refuse(lambda r: r.pop("source_to_state_correspondence"))
    _refuse(lambda r: r.pop("retention_evidence"))
    _refuse(lambda r: r.pop("state_mass_fields"))
    _refuse(lambda r: r["source_to_state_correspondence"].pop("QG"))
    _refuse(lambda r: r["state_mass_fields"]["qg"].pop("nonzero_mask_sha256"))
    _refuse(lambda r: r["state_mass_fields"]["qg"].pop("decoded_source"))
    # A species claimed PROVEN on a source that carried no mass.
    _refuse(
        lambda r: r["retention_evidence"]["QG"].update(
            {"source_nonzero_count": 0}),
        match="contradicts its own source mass count")
    # Evidence that disagrees with the fingerprints beside it.
    _refuse(
        lambda r: r["retention_evidence"]["QG"].update(
            {"state_nonzero_count": 1}),
        match="disagrees with its own fingerprints")
    # The headline B-H1 failure: every nonzero analyzed cell lost.
    def _lose_all_graupel(receipt):
        receipt["state_mass_fields"]["qg"]["nonzero_count"] = 0
        receipt["retention_evidence"]["QG"]["state_nonzero_count"] = 0
    _refuse(_lose_all_graupel, match="lost every nonzero analyzed QG")


def test_the_consumer_reports_a_vacuous_retention_claim_without_refusing_it():
    """A cloud-free analysis is a legitimate preparation, and says so.

    The wrapper must not pass a vacuous receipt off as proof, and must not
    refuse it either -- a genuinely cloud-free domain prepares correctly.
    """

    empty = _decoded_native_hrrr_initialization(18, analyzed_species=())
    receipt = _initial_hrrr_microphysics_receipt(
        empty.state, NSSL2_PROFILE_ID, empty.hydrometeor_initialization)
    summary = prepare._validated_retention_evidence(receipt)
    assert summary["strength"] == "VACUOUS"
    assert summary["vacuous_species"] == ["QC", "QG", "QI", "QR", "QS"]

    proven = prepare._validated_retention_evidence(
        _producer_receipt(NSSL2_PROFILE_ID))
    assert proven["strength"] == "PROVEN"
    assert proven["vacuous_species"] == []


def test_public_wrapper_accepts_64_decoder_workers_independent_of_native_budget():
    report = {
        "status": "PASS",
        "preparation": {
            "preprocess_backend": {"backend": "cuda"},
            "preprocess_worker_budget": {
                "schema": "gpuwm-preprocess-worker-budget-v1",
                "backend": "cuda",
                "applicable": False,
                "pipeline_decoder_workers_included": False,
                "peak_active_native_workers": 0,
            },
        },
        "pipeline": {"workers": {"requested": "64", "selected": 64}},
    }

    _preprocess, native_budget, decoder = prepare._validated_worker_receipts(
        report, selected_backend="cuda", requested_preprocess_workers=None,
        requested_pipeline_workers="64", final_hour=12)

    assert decoder["selected"] == 64
    assert native_budget["pipeline_decoder_workers_included"] is False


@pytest.mark.parametrize(
    "physics_profile",
    (
        KESSLER_PROFILE_ID,
        WSM6_PROFILE_ID,
        THOMPSON_PROFILE_ID,
        MORRISON_PROFILE_ID,
        NSSL2_PROFILE_ID,
        NSSL2_LEGACY_RRTMG_PROFILE_ID,
    ),
)
def test_public_wrapper_preserves_absolute_source_leads_and_rebases_model_time(
        tmp_path: Path, monkeypatch, physics_profile: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for hour in (12, 13):
        (source / f"hrrr.t05z.wrfnatf{hour:02d}.grib2").write_bytes(
            f"atmos-{hour}".encode("ascii"))
        (source / f"hrrr.t05z.soilf{hour:02d}.grib2").write_bytes(
            f"soil-{hour}".encode("ascii"))
    source_manifest = source / "SHA256SUMS"
    source_manifest.write_text("fixture\n", encoding="ascii")
    static_cache = tmp_path / "static.npz"
    static_receipt = tmp_path / "static.json"
    namelist = tmp_path / "namelist.input"
    for path in (static_cache, static_receipt, namelist):
        path.write_bytes(b"fixture")
    decoder = tmp_path / "hrrr_grib2_bridge"
    decoder.write_bytes(b"decoder")
    monkeypatch.setattr(prepare, "_decoder", lambda _env: decoder)

    commands: list[list[str]] = []

    def fake_run(command: list[str], _env: dict[str, str]) -> None:
        commands.append(command)
        if any(value.endswith("hrrr_single_domain_benchmark.py")
               for value in command):
            outdir = Path(command[command.index("--outdir") + 1])
            outdir.mkdir(parents=True)
            (outdir / "report.json").write_text(json.dumps({
                "status": "PASS",
                "history_interval_seconds": 3600.0,
                "physics": {
                    "schema": "gpuwm-prepared-physics-profile-v1",
                    "profile": physics_profile,
                    "hrrr_initialization": _cold_start_receipt(
                        physics_profile),
                },
                "preparation": {
                    "preprocess_backend": {"backend": "cuda"},
                    "preprocess_worker_budget": {
                        "schema": "gpuwm-preprocess-worker-budget-v1",
                        "backend": "cuda",
                        "applicable": False,
                        "pipeline_decoder_workers_included": False,
                        "peak_active_native_workers": 0,
                    },
                },
                "pipeline": {"workers": {"requested": "8", "selected": 8}},
            }), encoding="utf-8")

    monkeypatch.setattr(prepare, "_run", fake_run)
    output = tmp_path / "output"
    assert prepare.main([
        "--source-root", str(source),
        "--source-manifest", str(source_manifest),
        "--source-manifest-sha256", "0" * 64,
        "--static-cache", str(static_cache),
        "--static-receipt", str(static_receipt),
        "--namelist-input", str(namelist),
        "--physics-profile", physics_profile,
        "--valid-time", "2026-07-18_05:00:00",
        "--forecast-start-hour", "12",
        "--forecast-end-hour", "13",
        "--run-seconds", "3600",
        "--history-interval-seconds", "3600",
        "--output-root", str(output),
    ]) == 0

    series = output / "native" / "hrrr-f12-f13-series.tsv"
    assert [line.split("\t", 1)[0]
            for line in series.read_text(encoding="utf-8").splitlines()] \
        == ["12", "13"]
    benchmark = next(
        command for command in commands
        if any(value.endswith("hrrr_single_domain_benchmark.py")
               for value in command))
    assert benchmark[benchmark.index("--forecast-start-hour") + 1] == "12"
    assert benchmark[benchmark.index("--forecast-end-hour") + 1] == "13"
    assert benchmark[benchmark.index("--physics-profile") + 1] \
        == physics_profile
    assert benchmark[benchmark.index("--history-interval-seconds") + 1] \
        == "3600.0"
    assert "--io-mode" not in benchmark
    export = next(command for command in commands if "gpuwm.wrf_direct" in command)
    assert export[export.index("--valid-time") + 1] == "2026-07-18_17:00:00"

    receipt = json.loads(
        (output / "public-wrapper-result.json").read_text(encoding="utf-8"))
    assert receipt["source_cycle"] == "2026-07-18T05:00:00"
    assert receipt["model_start_time"] == "2026-07-18T17:00:00"
    assert receipt["source_forecast_hours"] == [12, 13]
    assert receipt["model_forcing_hours"] == [0, 1]
    assert receipt["forcing_hours"] == [0, 1]
    assert receipt["history_interval_seconds"] == 3600.0
    assert receipt["physics"]["profile"] == physics_profile


def test_public_wrapper_rejects_mismatched_or_incomplete_physics_receipt():
    complete = {
        "physics": {
            "schema": "gpuwm-prepared-physics-profile-v1",
            "profile": MORRISON_PROFILE_ID,
            "hrrr_initialization": _cold_start_receipt(
                MORRISON_PROFILE_ID),
        },
    }
    assert prepare._validated_physics_receipt(
        complete, requested_profile=MORRISON_PROFILE_ID) \
        == complete["physics"]

    with pytest.raises(RuntimeError, match="differs from the request"):
        prepare._validated_physics_receipt(
            complete, requested_profile=NSSL2_PROFILE_ID)
    incomplete = json.loads(json.dumps(complete))
    del incomplete["physics"]["hrrr_initialization"][
        "state_source_absent_fields"]
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            incomplete, requested_profile=MORRISON_PROFILE_ID)

    wrong_fields = json.loads(json.dumps(complete))
    wrong_fields["physics"]["hrrr_initialization"][
        "source_absent_wrf_fields"] = []
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            wrong_fields, requested_profile=MORRISON_PROFILE_ID)

    false_exact = json.loads(json.dumps(complete))
    false_exact["physics"]["hrrr_initialization"][
        "state_source_absent_fields"]["ng"]["all_exact_expected"] = False
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            false_exact, requested_profile=MORRISON_PROFILE_ID)

    wrong_bits = json.loads(json.dumps(complete))
    wrong_bits["physics"]["hrrr_initialization"][
        "state_source_absent_fields"]["ng"]["expected_uint32_bits"] = 1
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            wrong_bits, requested_profile=MORRISON_PROFILE_ID)
