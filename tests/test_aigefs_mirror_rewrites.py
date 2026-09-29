"""The AI-GEFS mirror's re-encoded archive prepares, and nothing else does.

The AWS mirror serves part of the AI-GEFS archive under another GRIB
writer's octets.  From 2026-01-11 00Z to 2026-04-08 06Z every message
carries PDT 0 (no ensemble octets), master table 4, local table 0 and
generating process 255; from 2026-04-08 12Z to 2026-04-24 12Z the
ensemble octets survive (PDT 1/11) but the generating process is 255.
Before these declarations the mapping pinned PDT 1 on every selector,
so a whole PDT 0 cycle matched nothing ("0 of 177 GRIB message(s) ...
match this mapping's selectors"), and the member verifier refused both
forms before any decode.

The fixtures are whole production envelopes of those cycles (provenance
and SHA-256 in the corpus README): the 100 m wind pair of the 2026-01-20
00Z mem000 and mem001 sfc f000, which are the same bytes in both members
because nothing in them names a member, the mem000 file's mean-sea-level
pressure record (centre 7 where its neighbours carry 74), and the 100 m
wind pair of 2026-04-10 00Z, whose perturbationNumber survived.  Every
fact is read through the real Rust inventory and both mapped engines.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil

import pytest

from gpuwm import bridges, mapped_engine_bridge
from gpuwm.mapped_source import (_grib_selectors_overlap, inspect_mapped_source,
                                 load_mapping)
from gpuwm.member_grammar import MemberIdentityRefusal, load_member_grammar
from gpuwm.member_prep import RECEIPT_NAME, prepare_member, verify_member_file
from gpuwm.source_authorities import (packaged_authorities,
                                      packaged_member_grammar)

_TESTS = Path(__file__).resolve().parent
FIXTURES = _TESTS / "fixtures" / "ensemble-member-identity"
PROFILE_ID = "aigefs-member-hybrid-grib2-v1"
MEMBER_SET = "aigefs-ensemble-grib2-members-v1"
WITHOUT_OCTETS = "mirror-rewrite-without-ensemble-octets"
WITHOUT_PROCESS = "mirror-rewrite-without-generating-process"
REWRITE_FORM = {"pdt": 0, "master_table_version": 4, "local_table_version": 0}
DETERMINISTIC = _TESTS / "fixtures" / "gdas-process-id" / (
    "nomads-gdas-20260729t12z-f000.grib2")


def _aigefs():
    return load_member_grammar(packaged_member_grammar(MEMBER_SET))


def _sfc(day: str, member: str) -> Path:
    return (FIXTURES / f"aigefs.{day}" / "00" / member / "model" / "atmos"
            / "grib2" / "aigefs.t00z.sfc.f000.grib2")


# ---------------------------------------------------------------------------
# The mapping: one ranked second form per selector
# ---------------------------------------------------------------------------

BASE = {"format": "grib2", "discipline": 0, "category": 3, "parameter": 1,
        "level_type": 101}


def test_selectors_pinning_different_pdts_are_disjoint():
    """One GRIB2 record carries one product definition template, so a
    field may rank a PDT 1 selector above a PDT 0 form of the same
    record without the two claiming one record twice."""

    assert not _grib_selectors_overlap(
        {**BASE, "pdt": 1}, {**BASE, "pdt": 0}, "grib2")
    assert _grib_selectors_overlap(BASE, {**BASE, "pdt": 0}, "grib2")
    assert _grib_selectors_overlap(
        {**BASE, "pdt": 1}, {**BASE, "pdt": 1}, "grib2")


def test_every_selector_ranks_the_producer_form_before_the_rewrite_form():
    mapping = load_mapping(packaged_authorities(PROFILE_ID)["mapping"])
    selecting = {name: field["selectors"]
                 for name, field in mapping["fields"].items()
                 if field.get("selectors")}
    assert "air_pressure_at_mean_sea_level" in selecting
    for name, selectors in selecting.items():
        assert len(selectors) == 2, name
        producer, rewrite = selectors
        assert producer["pdt"] == 1, name
        assert not {"master_table_version", "local_table_version"} & set(
            producer), name
        # The same record, under the rewrite's writer.  NCEP deterministic
        # products stamp master table 2 and local table 1, so a GFS, GDAS
        # or AIGFS file claimed as a member still matches nothing.
        assert rewrite == {**producer, **REWRITE_FORM}, name


def _decoders() -> dict[str, Path]:
    found = {name: bridges.find_bridge(name)
             for name in ("grib2_inventory", "grib2_dump")}
    if any(path is None for path in found.values()):
        pytest.skip("grib2 decoder executables absent")
    return found


@pytest.fixture(params=["python", "rust"])
def engine(request, monkeypatch):
    """Both engines decode the rewritten record, each through its own
    selector matcher."""

    if request.param == "python":
        return _decoders()
    if mapped_engine_bridge.find_engine() is None:
        pytest.skip("mapped engine not built in this checkout")
    monkeypatch.setenv(mapped_engine_bridge.ENGINE_ENV,
                       mapped_engine_bridge.ENGINE_RUST)
    return {}


def test_the_rewritten_sea_level_pressure_record_decodes(engine):
    """Before the rewrite form, all three messages matched no selector
    and the decode refused; now the mean-sea-level pressure record is
    read (under its centre 7, which the form does not pin) and the
    100 m winds, which no field maps, stay unread."""

    report = inspect_mapped_source(
        packaged_authorities(PROFILE_ID)["mapping"],
        [_sfc("20260120", "mem000")], **engine)
    decoded = {name for frame in report["frames"]
               for name in frame["decoded_direct_fields"]}
    assert decoded == {"air_pressure_at_mean_sea_level"}, report


def test_a_deterministic_record_still_matches_no_selector(engine):
    with pytest.raises(ValueError, match="match this mapping's selectors"):
        inspect_mapped_source(
            packaged_authorities(PROFILE_ID)["mapping"], [DETERMINISTIC],
            **engine)


# ---------------------------------------------------------------------------
# The member grammar: both rewrites verify, by the identity they carry
# ---------------------------------------------------------------------------

def test_the_rewrite_without_ensemble_octets_verifies_by_its_member_path():
    for member in ("mem000", "mem001"):
        evidence = verify_member_file(
            _aigefs(), member, _sfc("20260120", member))
        assert evidence.encoding == WITHOUT_OCTETS
        assert evidence.member_identity == "path"
        assert evidence.product_definition_templates == (0,)
        assert evidence.type_of_ensemble_forecast is None
        assert evidence.type_of_generating_process == (255,)


def test_octetless_bytes_under_another_members_path_refuse():
    """The mem001 fixture is the same bytes as the head of mem000's, so
    the path is the only thing that can say which member it is."""

    with pytest.raises(MemberIdentityRefusal) as caught:
        verify_member_file(_aigefs(), "mem002", _sfc("20260120", "mem001"))
    message = str(caught.value)
    assert WITHOUT_OCTETS in message
    assert "'mem002'" in message and "path component" in message


def test_the_rewrite_that_kept_its_octets_still_verifies_the_ordinal():
    evidence = verify_member_file(_aigefs(), "mem001", _sfc("20260410", "mem001"))
    assert evidence.encoding == WITHOUT_PROCESS
    assert evidence.member_identity == "ordinal"
    assert evidence.perturbation_number == 1
    with pytest.raises(MemberIdentityRefusal, match="perturbationNumber 1"):
        verify_member_file(_aigefs(), "mem000", _sfc("20260410", "mem001"))


def test_deterministic_bytes_under_a_member_path_keep_the_producer_refusal(
        tmp_path):
    """A GDAS file (master table 2, local table 1) placed where a member
    lives is no declared writer, so the producer's own refusal stands."""

    placed = tmp_path / "mem000" / DETERMINISTIC.name
    placed.parent.mkdir()
    shutil.copyfile(DETERMINISTIC, placed)
    with pytest.raises(MemberIdentityRefusal) as caught:
        verify_member_file(_aigefs(), "mem000", placed)
    message = str(caught.value)
    assert "no ensemble identity octets at all" in message
    assert "rewrite" not in message


def test_the_member_step_stages_a_rewritten_member_and_names_its_encoding(
        tmp_path):
    """The step every prepared chain runs before preparation: the staged
    tree keeps the member component, and its receipt says which declared
    encoding the bytes verified under."""

    member_dir = prepare_member(
        grammar_id=MEMBER_SET, member_id="mem000",
        cycle=datetime(2026, 1, 20, 0), steps=[0], inputs_root=FIXTURES,
        output_root=tmp_path, products=["sfc"])
    receipt = json.loads((member_dir / RECEIPT_NAME).read_text("utf-8"))
    [row] = receipt["files"]
    assert row["observed"]["encoding"] == WITHOUT_OCTETS
    assert row["observed"]["member_identity"] == "path"
    staged = member_dir / row["staged"]
    assert "mem000" in staged.parts
    # The chain verifies the staged copy again before preparation reads it.
    assert verify_member_file(_aigefs(), "mem000", staged).encoding == (
        WITHOUT_OCTETS)


def test_the_rust_member_table_is_the_packaged_grammar():
    """The Rust fetch and ingest verifier reads its own copy of each
    member grammar; a copy that drifted would accept bytes the
    preparation refuses, or refuse bytes it accepts."""

    tables = (_TESTS.parent / "tools" / "rustwx" / "crates" / "rustwx-models"
              / "src" / "member_tables")
    for name, member_set in (
            ("aigefs.json", MEMBER_SET),
            ("gefs.json", "gefs-ensemble-grib2-members-v1")):
        assert (tables / name).read_bytes() == packaged_member_grammar(
            member_set).read_bytes(), name
