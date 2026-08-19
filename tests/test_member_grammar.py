"""The ensemble member-addressing grammar: declaration, resolution, refusals.

The capability under test is generic: nothing in the engine knows a
model's name, and everything model-shaped -- the unflagged AIGEFS
control, GEFS's control-excluding ensemble-size octet, member-as-path-
component addressing -- must be expressible as rows of a packaged
``rw-wps.members.v1`` document.
"""

from __future__ import annotations

from datetime import datetime
import json

import pytest

from gpuwm.member_grammar import (MemberGrammar, MemberGrammarError,
                                  MemberIdentity, MemberIdentityRefusal,
                                  StatisticIdentity, load_member_grammar)
from gpuwm.source_authorities import (packaged_member_grammar,
                                      packaged_member_grammar_ids)


def _document(**overrides):
    """A minimal valid members document, synthetic, no model names."""

    document = {
        "schema": "rw-wps.members.v1",
        "name": "synthetic-members",
        "format": "grib2",
        "layout": "file_per_member",
        "declared_member_count": 4,
        "classes": {
            "control": {
                "ordinals": [0],
                "member_id": "c00",
                "token": "xc00",
                "verification": {
                    "product_definition_templates": [1, 11],
                    "type_of_ensemble_forecast": 1,
                    "perturbation_number": "ordinal",
                    "ensemble_size": 3,
                },
            },
            "perturbed": {
                "ordinals": {"first": 1, "last": 3},
                "member_id": "p{ordinal:02d}",
                "token": "xp{ordinal:02d}",
                "verification": {
                    "product_definition_templates": [1, 11],
                    "type_of_ensemble_forecast": 3,
                    "perturbation_number": "ordinal",
                    "ensemble_size": 3,
                },
            },
        },
        "statistics": {
            "xavg": {
                "statistic": "ensemble mean",
                "token": "xavg",
                "product_definition_templates": [2, 12],
                "derived_forecast": 0,
            },
        },
        "products": {
            "main": {
                "relative_path":
                    "cycle.{yyyymmdd}/{hh}/{token}.t{hh}z.f{fff}",
            },
        },
    }
    document.update(overrides)
    return document


def _grammar(**overrides) -> MemberGrammar:
    return MemberGrammar(_document(**overrides), source="synthetic")


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------

def test_the_declared_member_set_enumerates_ids_ordinals_and_tokens():
    grammar = _grammar()
    members = {member.member_id: member for member in grammar.members()}
    assert sorted(members) == ["c00", "p01", "p02", "p03"]
    assert members["c00"].ordinal == 0
    assert members["c00"].token == "xc00"
    assert members["p03"].token == "xp03"
    assert members["p03"].verification.type_of_ensemble_forecast == 3


def test_a_count_the_classes_do_not_back_refuses():
    with pytest.raises(MemberGrammarError, match="declared_member_count"):
        _grammar(declared_member_count=31)


def test_duplicate_ordinals_across_classes_refuse():
    document = _document()
    document["classes"]["control"]["ordinals"] = [1]
    document["declared_member_count"] = 4
    with pytest.raises(MemberGrammarError, match="ordinal 1"):
        MemberGrammar(document, source="synthetic")


def test_an_unsupported_layout_refuses_by_name():
    with pytest.raises(MemberGrammarError, match="concatenated-members"):
        _grammar(layout="concatenated_members")


def test_a_non_ordinal_perturbation_rule_refuses():
    document = _document()
    document["classes"]["control"]["verification"][
        "perturbation_number"] = "ordinal_plus_one"
    with pytest.raises(MemberGrammarError, match="ordinal"):
        MemberGrammar(document, source="synthetic")


def test_unknown_keys_refuse_rather_than_silently_not_enforcing():
    with pytest.raises(MemberGrammarError, match="unknown keys"):
        _grammar(surprise="value")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_member_paths_resolve_from_cycle_step_and_token():
    grammar = _grammar()
    assert grammar.relative_path(
        "p02", "main", datetime(2026, 8, 17, 6), 3,
    ) == "cycle.20260817/06/xp02.t06z.f003"


def test_an_unknown_member_id_refusal_lists_the_declared_set():
    grammar = _grammar()
    with pytest.raises(MemberIdentityRefusal, match="c00"):
        grammar.member("p99")


def test_a_statistic_asked_for_as_a_member_refuses_by_name():
    grammar = _grammar()
    with pytest.raises(MemberIdentityRefusal) as caught:
        grammar.member("xavg")
    message = str(caught.value)
    assert "ensemble mean" in message
    assert "not a member" in message


def test_classification_tells_members_and_statistics_apart():
    grammar = _grammar()
    member = grammar.classify_relative_path(
        "cycle.20260817/06/xp02.t06z.f003")
    assert isinstance(member, MemberIdentity)
    assert member.member_id == "p02"
    statistic = grammar.classify_relative_path(
        "cycle.20260817/06/xavg.t06z.f003")
    assert isinstance(statistic, StatisticIdentity)
    assert statistic.statistic == "ensemble mean"
    assert grammar.classify_relative_path("elsewhere/file") is None


# ---------------------------------------------------------------------------
# The packaged documents
# ---------------------------------------------------------------------------

def test_every_packaged_member_grammar_loads_under_the_schema():
    for grammar_id in packaged_member_grammar_ids():
        grammar = load_member_grammar(packaged_member_grammar(grammar_id))
        assert len(grammar.members()) == grammar.declared_member_count


def test_the_packaged_sets_declare_the_measured_control_conventions():
    """The two shipped ensembles disagree about their own controls.

    One flags its control low-resolution control (type 1) and encodes an
    ensemble size that EXCLUDES it; the other flags its control exactly
    like a perturbed member (type 3) and encodes a size that INCLUDES
    it.  Both facts are table data; if either moved into code, the other
    source would break.
    """

    conventions = set()
    for grammar_id in packaged_member_grammar_ids():
        grammar = load_member_grammar(packaged_member_grammar(grammar_id))
        control = grammar.member_for_ordinal(0)
        assert control is not None
        conventions.add((
            control.verification.type_of_ensemble_forecast,
            control.verification.ensemble_size == grammar.declared_member_count,
        ))
    assert conventions == {(1, False), (3, True)}


def test_path_component_member_identity_is_expressible_as_table_data():
    """A member that is a directory, not a filename token.

    The leaf filename must be identical for every member and the path
    still classify correctly -- the trap that makes flat downloads
    destroy member identity on such feeds.
    """

    for grammar_id in packaged_member_grammar_ids():
        grammar = load_member_grammar(packaged_member_grammar(grammar_id))
        members = grammar.members()
        cycle = datetime(2026, 8, 17, 0)
        for product in grammar.products():
            paths = [grammar.relative_path(m.member_id, product, cycle, 0)
                     for m in members]
            assert len(set(paths)) == len(members)
            for member, path in zip(members, paths):
                assert grammar.classify_relative_path(path) is not None
                found = grammar.classify_relative_path(path)
                assert isinstance(found, MemberIdentity)
                assert found.member_id == member.member_id


def test_a_broken_document_refuses_with_its_source_named(tmp_path):
    path = tmp_path / "broken.members.json"
    path.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
    with pytest.raises(MemberGrammarError, match="broken.members.json"):
        load_member_grammar(path)
