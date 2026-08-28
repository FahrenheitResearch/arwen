"""The mp=28 aerosol source reaches report.json, and only where it applies.

TWO CLAIMS, and the second is the one with teeth.

  1. A real aerosol-aware Thompson preparation's report names WHICH source
     initialized nwfa/nifa, and pins the dataset it read by content --
     path, byte count and computed SHA-256 -- so two runs against two WRF
     releases' copies of QNWFA_QNIFA_SIGMA_MONTHLY.dat stay tellable apart
     afterwards.  Before this, the only place that fact was ever spoken
     was the preparing process's stderr.

  2. A run under a scheme that has no water-friendly/ice-friendly aerosol
     fields gains NO key.  Not ``dataset_used: false``, not
     ``aerosol_source: null``, nothing.  "This scheme has no such fields"
     and "this mp=28 run could not find the dataset and ran on an analytic
     profile" are different facts, and a key that renders both as an
     absence puts a WSM6 forecast and a fallen-back aerosol-aware forecast
     in the same bucket -- which is exactly the pair a reader consults
     this receipt to tell apart.

The suite is CPU-only.  The climatology test needs WRF's 225 MB dataset,
which this tree does not redistribute, and locates it the way the PRODUCT
does (``resolve_wif_climatology``) so it cannot pass on a path the
shipping code would not have found; it skips where there is no copy.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import numpy as np
import pytest

from gpuwm.aerosol_source_receipt import (
    AEROSOL_SOURCE_BY_DOMAIN_KEY,
    AEROSOL_SOURCE_KEY,
    AEROSOL_SOURCE_SCHEMA,
    aerosol_source_report_entries,
    aerosol_source_report_entry,
    scheme_has_aerosol_number_fields,
)
from gpuwm.ingest.prepared_cache import write_prepared_cache

# The real ingest, and the real dataset fixture, rather than a second
# spelling of either: what this suite asserts about a report has to be
# what a user's run puts in one.
from test_prepared_cache import _fixture as _prepared_cache_fixture
from test_real_init import (  # noqa: F401 - wif_dataset is a fixture
    _analyzed_hrrr_real_init,
    wif_dataset,
)


#: The five schemes this tree's real routes actually resolve to besides
#: aerosol-aware Thompson: Kessler, WSM6, classic Thompson, Morrison and
#: NSSL-2.  None of them declares qnwfa/qnifa, so none of them may gain a
#: key.  Listed rather than sampled so a scheme cannot quietly join the
#: aerosol-aware set without this failing.
_SCHEMES_WITHOUT_AEROSOL_FIELDS = (1, 6, 8, 10, 18)


def _mp28_climatology_receipt(monkeypatch, dataset):
    """One real mp=28 initialization that READ the dataset."""

    monkeypatch.setenv("GPUWM_WIF_CLIMATOLOGY", str(dataset))
    # The geodesy a GLOBAL monthly dataset needs, supplied the way the
    # production routes supply it (through ``grid=``, derived inside
    # initialize_real) rather than through any test-only hook.
    lat2d = np.array([[39.4, 39.4, 39.4], [39.6, 39.6, 39.6]])
    lon2d = np.array([[-98.7, -98.5, -98.3], [-98.7, -98.5, -98.3]])
    result, cfg = _analyzed_hrrr_real_init(
        28, wif_grid_latlon=(lat2d, lon2d),
        wif_valid_date="2026-08-25T18:00:00")
    return result, cfg


def _empty_wif_search(monkeypatch, tmp_path):
    """Guarantee the fallback: every rung of the resolver comes up empty.

    All four rungs, not two.  ``resolve_wif_climatology`` searches the
    config path, ``$GPUWM_WIF_CLIMATOLOGY``, ``$GPUWM_WIF_CLIMATOLOGY_
    ROOT``, the root ``gpuwm fetch-tables --wif`` stages into, and then
    the working directory.  On the reference host the staged root HOLDS
    the dataset, so unsetting the two overrides alone would resolve the
    real copy and this test would pass -- or fail -- for a reason it is
    not about.
    """
    from gpuwm.ingest import wif_climatology

    monkeypatch.delenv(
        wif_climatology.WIF_CLIMATOLOGY_PATH_ENV, raising=False)
    monkeypatch.delenv(
        wif_climatology.WIF_CLIMATOLOGY_ROOT_ENV, raising=False)
    monkeypatch.setenv("GPUWM_WIF_DATA_ROOT", str(tmp_path / "no-staged-wif"))
    monkeypatch.chdir(tmp_path)


def test_a_real_mp28_preparation_reports_its_dataset_choice_and_identity(
        monkeypatch, tmp_path, wif_dataset):  # noqa: F811 - imported fixture
    """CLAIM 1, end to end: ingest -> prepared cache -> report block.

    The whole carriage, because every link was where the fact used to be
    lost.  ``initialize_real`` decides; ``write_prepared_cache`` has to
    store the decision, since the process that publishes report.json is a
    different process reading the bundle back cold; and the report block
    has to name the choice and the bytes.
    """
    result, cfg = _mp28_climatology_receipt(monkeypatch, wif_dataset)

    # The ingest chose the climatology and filled the fields itself.
    assert result.aerosol_initialization["aerosol_source"] == (
        "wif-climatology")

    # THE CACHE CARRIES IT.  Written through the real writer with a real
    # RealInitResult receipt attached to the fixture's initial result.
    initial, met, boundaries = _prepared_cache_fixture()
    initial.aerosol_initialization = dict(result.aerosol_initialization)
    write_prepared_cache(
        tmp_path / "cache", identity={"source": "aerosol-source-report-test"},
        initial_result=initial, met=met, boundaries=boundaries)
    header = json.loads(
        (tmp_path / "cache" / "header.json").read_text(encoding="utf-8"))
    stored = header["metadata"][AEROSOL_SOURCE_KEY]
    assert stored["aerosol_source"] == "wif-climatology"

    # THE REPORT NAMES THE CHOICE, from the cache metadata exactly as
    # gpuwm/prepared_single_domain_forecast.py reads it -- through a
    # read-only mapping, which is what a restored cache hands back.
    report = {"schema": "test-report-v1", "status": "PASS"}
    report.update(aerosol_source_report_entry(
        MappingProxyType(stored), mp_physics=cfg.mp_physics,
        when_unrecorded="unused: the receipt is present"))
    entry = report[AEROSOL_SOURCE_KEY]
    assert entry["schema"] == AEROSOL_SOURCE_SCHEMA
    assert entry["recorded"] is True
    assert entry["dataset_used"] is True
    assert entry["aerosol_source"] == "wif-climatology"
    assert entry["synthetic_fallback_in_use"] is False

    # THE PINNED IDENTITY, and it is CONTENT-ADDRESSED rather than a
    # repeated constant: the report's digest must be the digest of the
    # bytes this run actually opened, which is the only form of the claim
    # that can distinguish two WRF releases' copies.
    dataset = entry["dataset"]
    assert dataset["resolved"] is True
    assert dataset["path"] == str(wif_dataset)
    assert dataset["bytes"] == wif_dataset.stat().st_size
    digest = hashlib.sha256()
    with open(wif_dataset, "rb") as stream:
        while block := stream.read(1 << 22):
            digest.update(block)
    assert dataset["sha256"] == digest.hexdigest()
    assert isinstance(dataset["matches_wrf_471_reference"], bool)

    # And the whole thing survives the trip a report actually takes.
    assert json.loads(json.dumps(report))[AEROSOL_SOURCE_KEY] == entry


@pytest.mark.parametrize("mp_physics", _SCHEMES_WITHOUT_AEROSOL_FIELDS)
def test_a_scheme_without_aerosol_fields_gains_no_key(mp_physics):
    """CLAIM 2.  Not a null, not a false -- nothing.

    Asserted on the WHOLE report rather than on the one key, because the
    failure this prevents is a reader seeing any aerosol-shaped field on a
    WSM6 receipt and concluding something about a dataset.
    """
    assert scheme_has_aerosol_number_fields(mp_physics) is False
    before = {"schema": "test-report-v1", "status": "PASS", "physics": {}}
    report = dict(before)
    report.update(aerosol_source_report_entry(
        {}, mp_physics=mp_physics,
        when_unrecorded="must never be reached for these schemes"))
    assert report == before
    assert AEROSOL_SOURCE_KEY not in report
    assert "aerosol" not in json.dumps(report).lower()


def test_a_real_non_aerosol_ingest_receipt_produces_no_key():
    """The same claim through the real ingest rather than an empty dict.

    ``initialize_real`` returns an empty ``aerosol_initialization`` for
    every scheme but 28.  This runs one and feeds exactly what it returned
    into the report builder, so the test cannot pass on a hand-made input
    the product never produces.
    """
    result, cfg = _analyzed_hrrr_real_init(8)
    assert result.aerosol_initialization == {}
    report = {"status": "PASS"}
    report.update(aerosol_source_report_entry(
        result.aerosol_initialization, mp_physics=cfg.mp_physics,
        when_unrecorded="must never be reached for classic Thompson"))
    assert report == {"status": "PASS"}


def test_the_fallback_and_the_inapplicable_scheme_are_not_the_same_absence(
        monkeypatch, tmp_path):
    """THE DISTINCTION, asserted as a difference rather than as two facts.

    An mp=28 run that searched for the dataset and found none DID run
    without aerosol data: that is reported loudly, with the reason the
    search failed.  A WSM6 run reports nothing at all.  A reader holding
    the two receipts must be able to separate them, and this is the
    comparison that proves they can.
    """
    _empty_wif_search(monkeypatch, tmp_path)
    result, cfg = _analyzed_hrrr_real_init(28)
    assert cfg.mp_physics == 28

    fallback = {}
    fallback.update(aerosol_source_report_entry(
        result.aerosol_initialization, mp_physics=cfg.mp_physics,
        when_unrecorded="unused: the receipt is present"))
    inapplicable = {}
    inapplicable.update(aerosol_source_report_entry(
        {}, mp_physics=6, when_unrecorded="unused: no aerosol fields"))

    assert inapplicable == {}
    entry = fallback[AEROSOL_SOURCE_KEY]
    assert entry["recorded"] is True
    assert entry["dataset_used"] is False
    assert entry["synthetic_fallback_in_use"] is True
    assert entry["synthetic_fallback_requested"] is False
    assert entry["aerosol_source"] == "thompson_init-synthetic-profile"
    # It says WHY there was no dataset, concretely.  A fallback whose
    # reason is "unavailable" is a fallback nobody can act on.
    assert entry["dataset"]["resolved"] is False
    assert "QNWFA_QNIFA_SIGMA_MONTHLY.dat" in (
        entry["dataset"]["fallback_reason"])
    assert entry["dataset"]["candidates"]


def test_an_unrecorded_receipt_is_not_reported_as_no_dataset():
    """THE THIRD STATE, kept out of the second.

    A writer that holds no ingest receipt for an aerosol-aware run -- an
    offline child fed from an archived parent frame, a prepared cache
    written before the receipt was stored -- knows the question applies
    and does not know the answer.  Writing ``dataset_used: false`` there
    would be a claim it cannot make, so the key is absent and the block
    says instead what this route does know.
    """
    entry = aerosol_source_report_entry(
        {}, mp_physics=28,
        when_unrecorded="inherited from the parent history frames",
    )[AEROSOL_SOURCE_KEY]
    assert entry["recorded"] is False
    assert "dataset_used" not in entry
    assert "aerosol_source" not in entry
    assert entry["not_recorded_because"] == (
        "inherited from the parent history frames")
    assert entry["scheme_has_aerosol_number_fields"] is True


def test_every_report_writer_that_carries_the_key_uses_the_shared_builder():
    """One builder, not six spellings of the rule.

    The applicability rule is the whole point of this module: a writer
    that re-implemented "if mp_physics == 28" locally would be a second
    place for the two absences to drift apart.  So the gate is that no
    report writer names the key without importing the builder.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    writers = (
        "gpuwm/prepared_single_domain_forecast.py",
        "gpuwm/prepared_domain_tree_forecast.py",
        "gpuwm/offline_child_run.py",
        "gpuwm/offline_child_smoke.py",
        "gpuwm/hrrr_hierarchy_direct.py",
        "tools/hrrr_single_domain_benchmark.py",
        "tools/hrrr_state_proof.py",
    )
    for name in writers:
        source = (repo / name).read_text(encoding="utf-8")
        assert "aerosol_source_report_entr" in source, name
        assert f'"{AEROSOL_SOURCE_KEY}":' not in source, (
            f"{name} spells the report key by hand; use the shared builder")
        assert f'"{AEROSOL_SOURCE_BY_DOMAIN_KEY}":' not in source, name


def test_the_three_late_wired_routes_take_the_one_shared_resolver():
    """TASK TWO's standing claim, as a gate rather than a commit message.

    Two of the three initialization routes that were never wired now pass
    ``grid=`` -- the same single front door the production real routes
    use, which is what makes ``resolve_wif_climatology`` the one resolver.
    The third is deliberately NOT wired and carries its reason inline; the
    thing that must never happen there is a SECOND resolver appearing to
    compensate.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]

    wired = (
        ("gpuwm/verify/cases/real74_d01.py",
         "horizontal, cfg, coord, terrain, grid=grid,"),
        ("tools/hrrr_state_proof.py",
         'met, cfg, coord, static["HGT_M"], grid=grid,'),
    )
    for name, call in wired:
        assert call in (repo / name).read_text(encoding="utf-8"), name

    benchmark = (repo / "tools/hrrr_single_domain_benchmark.py").read_text(
        encoding="utf-8")
    # The full-domain state IS wired.
    assert 'met, dc.run, coord, static["HGT_M"], grid=grid,' in benchmark
    # The boundary-strip construction is not, and says so by name.
    assert "NO ``grid=`` HERE, AND THAT IS THE DECISION" in benchmark
    # And nothing anywhere reintroduces the retired second resolver.
    for name in ("gpuwm/verify/cases/real74_d01.py",
                 "tools/hrrr_state_proof.py",
                 "tools/hrrr_single_domain_benchmark.py"):
        assert "resolve_wif_climatology_path" not in (
            repo / name).read_text(encoding="utf-8"), name


def test_the_prepared_cache_of_a_non_aerosol_run_is_unchanged(tmp_path):
    """The emptiness contract, measured.

    A cache written for a scheme with no aerosol receipt must canonicalize
    to the bytes it did before this row existed, or every stored
    ``prepared_header_sha256`` in the field stops matching.  Compared as
    two real caches: one written by an initial result carrying an EMPTY
    receipt, one carrying no such attribute at all.
    """
    plain_initial, met, boundaries = _prepared_cache_fixture()
    empty_initial, met2, boundaries2 = _prepared_cache_fixture()
    empty_initial.aerosol_initialization = {}
    assert not hasattr(plain_initial, "aerosol_initialization")

    first = write_prepared_cache(
        tmp_path / "plain", identity={"source": "x"},
        initial_result=plain_initial, met=met, boundaries=boundaries)
    second = write_prepared_cache(
        tmp_path / "empty", identity={"source": "x"},
        initial_result=empty_initial, met=met2, boundaries=boundaries2)
    assert first["content_sha256"] == second["content_sha256"]
    for path in (tmp_path / "plain", tmp_path / "empty"):
        header = json.loads(
            (path / "header.json").read_text(encoding="utf-8"))
        assert AEROSOL_SOURCE_KEY not in header["metadata"]


def test_the_prepared_cache_of_an_aerosol_run_carries_the_receipt(tmp_path):
    """And the row appears exactly when there is something to say."""
    initial, met, boundaries = _prepared_cache_fixture()
    initial.aerosol_initialization = {
        "aerosol_source": "thompson_init-synthetic-profile",
        "synthetic_fallback_in_use": True,
        "dataset": {"resolved": False, "fallback_reason": "none staged"},
    }
    write_prepared_cache(
        tmp_path / "cache", identity={"source": "x"},
        initial_result=initial, met=met, boundaries=boundaries)
    header = json.loads(
        (tmp_path / "cache" / "header.json").read_text(encoding="utf-8"))
    stored = header["metadata"][AEROSOL_SOURCE_KEY]
    assert stored["aerosol_source"] == "thompson_init-synthetic-profile"
    assert stored["dataset"]["resolved"] is False


def test_a_mapping_proxy_receipt_still_serializes(tmp_path):
    """A restored cache hands back read-only mappings all the way down.

    ``json.dumps`` serializes ``dict`` and not ``Mapping``, so a proxy
    anywhere in the tree would turn "write the report" into a TypeError
    naming nothing -- at the very end of a finished forecast.
    """
    proxy = MappingProxyType({
        "aerosol_source": "wif-climatology",
        "dataset": MappingProxyType({"resolved": True, "bytes": 1}),
        "not_initialized_here": ("nwfa", "nifa"),
    })
    entry = aerosol_source_report_entry(
        proxy, mp_physics=28, when_unrecorded="unused")[AEROSOL_SOURCE_KEY]
    text = json.dumps({"report": entry}, sort_keys=True)
    assert json.loads(text)["report"]["dataset"] == {
        "resolved": True, "bytes": 1}
    assert json.loads(text)["report"]["initialization_receipt"][
        "not_initialized_here"] == ["nwfa", "nifa"]


def test_the_helper_never_guesses_a_reason():
    """``when_unrecorded`` is required, at every call site, on purpose.

    There is no generic true sentence for "an aerosol-aware run whose
    receipt this writer does not hold" -- an offline child inherited its
    parent's aerosol, a prepared forecast is reading someone else's cache
    -- so the module refuses to invent one.
    """
    with pytest.raises(TypeError):
        aerosol_source_report_entry({}, mp_physics=28)


def test_the_state_proof_reports_what_it_resolves():
    """Resolving without reporting is how a receipt omits its own subject.

    ``tools/hrrr_state_proof.py`` gained the ``grid=`` front door; this
    pins that it also gained the report block, so an aerosol-aware
    configuration there cannot read a dataset and publish a proof that
    never mentions which one.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]
              / "tools/hrrr_state_proof.py").read_text(encoding="utf-8")
    assert "results[0].aerosol_initialization" in source
    assert "report.update(aerosol_source_report_entry(" in source


def test_nothing_here_is_a_second_resolution_path():
    """The one rule the previous integration bought with a deletion.

    A duplicate resolver over ``wif_climatology_path`` is how a run reads
    one dataset and reports another; the merge that unified this door
    deleted the second one.  Nothing added for the report may bring it
    back, so no report writer may import a resolver at all -- the choice
    is made once, at ingest, and carried.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    for name in ("gpuwm/aerosol_source_receipt.py",
                 "gpuwm/prepared_single_domain_forecast.py",
                 "gpuwm/prepared_domain_tree_forecast.py",
                 "gpuwm/offline_child_run.py",
                 "gpuwm/offline_child_smoke.py",
                 "gpuwm/hrrr_hierarchy_direct.py"):
        source = (repo / name).read_text(encoding="utf-8")
        assert "resolve_wif_climatology" not in source, name
        assert "resolve_wif_climatology_path" not in source, name


def test_the_builder_reports_the_configured_selector():
    """Which of the three ``mp28_aerosol_source`` values the run asked for.

    ``synthetic`` reached deliberately and ``synthetic`` reached because
    nothing was found produce the same fields and are not the same event;
    the receipt separates them and the report has to carry that through.
    """
    requested = aerosol_source_report_entry(
        {"aerosol_source": "thompson_init-synthetic-profile",
         "mp28_aerosol_source": "synthetic",
         "synthetic_fallback_in_use": True,
         "synthetic_fallback_requested": True},
        mp_physics=28, when_unrecorded="unused")[AEROSOL_SOURCE_KEY]
    assert requested["mp28_aerosol_source"] == "synthetic"
    assert requested["synthetic_fallback_requested"] is True
    assert requested["dataset_used"] is False


def test_a_non_integer_scheme_is_not_treated_as_aerosol_aware():
    """A report writer handing in something odd must not invent a key."""
    for value in (None, "", "twenty-eight", object()):
        assert scheme_has_aerosol_number_fields(value) is False
        assert aerosol_source_report_entry(
            {"aerosol_source": "wif-climatology"},
            mp_physics=value, when_unrecorded="unused") == {}


def test_the_sibling_receipt_still_rides_the_cache(tmp_path):
    """The row added beside it did not displace the one already there."""
    initial, met, boundaries = _prepared_cache_fixture()
    initial.hydrometeor_initialization = {"retained_correspondence": {}}
    initial.aerosol_initialization = {"aerosol_source": "wif-climatology"}
    write_prepared_cache(
        tmp_path / "cache", identity={"source": "x"},
        initial_result=initial, met=met, boundaries=boundaries)
    metadata = json.loads(
        (tmp_path / "cache" / "header.json").read_text(
            encoding="utf-8"))["metadata"]
    assert metadata["hydrometeor_initialization"] == {
        "retained_correspondence": {}}
    assert metadata[AEROSOL_SOURCE_KEY] == {
        "aerosol_source": "wif-climatology"}


def test_the_report_block_is_a_single_key(monkeypatch, tmp_path):
    """Merged into a report, it adds exactly one top-level key.

    The writers call ``report.update(...)``; a builder that returned two
    keys, or a nested-away one, would scatter the fact across receipts
    that a bundle tool has to find by name.
    """
    _empty_wif_search(monkeypatch, tmp_path)
    result, cfg = _analyzed_hrrr_real_init(28)
    before = {"schema": "test-report-v1", "status": "PASS"}
    report = dict(before)
    report.update(aerosol_source_report_entry(
        result.aerosol_initialization, mp_physics=cfg.mp_physics,
        when_unrecorded="unused"))
    assert set(report) - set(before) == {AEROSOL_SOURCE_KEY}


def test_the_receipt_travels_whole(monkeypatch, tmp_path):
    """The summary fields are a convenience; the receipt itself is kept.

    A reader who needs more than the four summary booleans -- the WRF
    citations, the per-field source-absent fingerprints, the stage receipt
    binding weights and operators -- must not have to go back to a process
    that has exited.
    """
    _empty_wif_search(monkeypatch, tmp_path)
    result, cfg = _analyzed_hrrr_real_init(28)
    entry = aerosol_source_report_entry(
        result.aerosol_initialization, mp_physics=cfg.mp_physics,
        when_unrecorded="unused")[AEROSOL_SOURCE_KEY]
    receipt = entry["initialization_receipt"]
    assert receipt["awaiting_profile_fill"] is True
    assert "module_mp_thompson.F" in receipt["microphysics_citation"]
    assert set(receipt["source_absent_state_fields"]) == {
        "nc", "nr", "ni", "nwfa", "nifa", "nwfa2d", "nifa2d"}
    assert isinstance(
        receipt["source_absent_state_fields"]["nwfa"]["sha256"], str)


def test_numpy_scalars_do_not_break_the_applicability_test():
    """``cfg.mp_physics`` arrives as an int, but a receipt round-trip can
    hand back a numpy integer; the rule must not silently fall through to
    "inapplicable" and drop the key on a genuine mp=28 run."""
    assert scheme_has_aerosol_number_fields(np.int32(28)) is True
    assert scheme_has_aerosol_number_fields(np.int64(6)) is False


def test_a_nest_tree_answers_per_domain_under_its_own_key():
    """A tree receipt keeps one answer per domain, and says so by name.

    One tree-wide verdict would lose the pair this exists to show: a root
    initialized from the climatology beside a child whose prepared cache
    predates the receipt.  And the multi-domain document takes a DIFFERENT
    key, because a consumer reading ``entry["dataset_used"]`` must not
    start throwing the moment it meets a tree.
    """
    entries = aerosol_source_report_entries(
        (("d01", 28, {"aerosol_source": "wif-climatology",
                      "dataset": {"resolved": True}}),
         ("d02", 28, {}),
         ("d03", 6, {})),
        when_unrecorded="the child's cache predates the receipt")
    blocks = entries[AEROSOL_SOURCE_BY_DOMAIN_KEY]
    # d03 is WSM6: no block at all, exactly as a single-domain WSM6 report
    # gains no key.
    assert set(blocks) == {"d01", "d02"}
    assert blocks["d01"]["dataset_used"] is True
    assert blocks["d02"]["recorded"] is False
    assert "dataset_used" not in blocks["d02"]
    assert AEROSOL_SOURCE_KEY not in entries


def test_a_tree_of_non_aerosol_domains_gains_nothing():
    """The emptiness contract holds for the multi-domain form too."""
    before = {"schema": "test-tree-receipt-v1", "status": "PASS"}
    report = dict(before)
    report.update(aerosol_source_report_entries(
        (("d01", 6, {}), ("d02", 10, {}), ("d03", 8, {})),
        when_unrecorded="must never be reached for these schemes"))
    assert report == before
    assert "aerosol" not in json.dumps(report).lower()


def test_both_key_names_are_found_by_one_grep():
    """Two shapes, two names, one substring -- so no run hides from a sweep."""
    assert AEROSOL_SOURCE_KEY in AEROSOL_SOURCE_BY_DOMAIN_KEY
    assert AEROSOL_SOURCE_KEY != AEROSOL_SOURCE_BY_DOMAIN_KEY
