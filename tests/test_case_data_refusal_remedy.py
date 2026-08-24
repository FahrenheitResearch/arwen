"""The ``[case_data]`` refusals must carry a remedy, and ``--explain``
must add substance rather than repeat the refusal.

Measured on this tree through the shipped CLI before the fix.  Both
refusals printed the SAME text with and without ``--explain``, and
neither named a way out:

  no table::

    gpuwm static: experiment config <path> carries no [case_data] table;
    the experiment runtime requires declared inputs (forcing, vtable,
    wps_namelist, geog_root, and policies).

  keys missing from a table that IS there::

    gpuwm static: [case_data] of <path> is missing required key(s)
    ['wps_namelist', 'geog_root', 'sfcp_to_sfcp']: every input path and
    policy is declared, never implicit.

A reader who has just been told which keys are required still does not
know what to write in them, which command writes them, or -- the case
that matters most, because it is the one a wizard emission actually
lands in -- that their config is not missing a table at all and belongs
on the other route.  Both messages were unlayered ``ValueError``s, so
the flag this product puts on every subcommand added exactly nothing.

Every test here fails against the tree as it stood at ``e8c05bdc0``.
"""

from __future__ import annotations

import pytest

from gpuwm.case_data import (_OPTIONAL_KEYS, _REQUIRED_KEYS, load_case_data,
                             load_experiment_case)
from gpuwm.explain import split

from test_case_data import _CASE_DATA_TOML, _EXPERIMENT_TOML, make_case_toml


def _write(tmp_path, *, tables: str = "", name: str = "case.toml"):
    """An experiment TOML with no ``[case_data]``, plus extra tables."""

    path = tmp_path / name
    path.write_text(_EXPERIMENT_TOML + tables, encoding="utf-8")
    return path


def _partial_case_toml(tmp_path, *, keep=("forcing", "vtable",
                                          "output_title")):
    """A ``[case_data]`` carrying only ``keep`` of the required keys."""

    lines = ["[case_data]"]
    for line in _CASE_DATA_TOML.strip().splitlines()[1:]:
        key = line.split("=")[0].strip()
        if key in keep:
            lines.append(line)
    return make_case_toml(tmp_path, case_data="\n".join(lines) + "\n")


# --------------------------------------------------------------------
# No [case_data] table at all
# --------------------------------------------------------------------

def _no_table_message(tmp_path, **kwargs) -> str:
    path = _write(tmp_path, **kwargs)
    with pytest.raises(ValueError) as caught:
        load_case_data(path)
    return str(caught.value)


def test_the_missing_table_refusal_names_the_command_that_writes_one(
        tmp_path):
    """The remedy is a command, spelled so it can be pasted."""

    action, _why = split(_no_table_message(tmp_path))
    assert "carries no [case_data] table" in action    # the breakage
    assert "remedy:" in action
    assert "gpuwm domain" in action                    # the door
    assert "--out" in action                           # ... with the flag
    assert "--cycle" in action                         # ... that door needs


def test_the_missing_table_refusal_names_the_keys_that_would_satisfy_it(
        tmp_path):
    """The other half of the remedy: what to write if writing by hand."""

    action, _why = split(_no_table_message(tmp_path))
    for key in _REQUIRED_KEYS:
        assert key in action, f"{key} is required and unnamed in the remedy"


def test_explain_adds_the_search_and_a_pasteable_table(tmp_path):
    """``--explain`` must add substance: what was looked for, where it
    was looked for, and a table that would satisfy the load."""

    action, why = split(_no_table_message(tmp_path))
    assert why, "the missing-table refusal must carry an --explain half"
    assert why != action
    # what was looked for, and where
    assert "looked for" in why.lower()
    assert "[case_data]" in why
    # ... and what the file DID carry, so the reader can see the mismatch
    assert "[experiment]" in why and "[shared]" in why
    # what would satisfy it: an assignment per required key, not a list
    # of names the reader has to go look up
    for key in _REQUIRED_KEYS:
        assert f"{key} =" in why, f"{key} has no pasteable line"
    # ... and the keys that are legal but not required, named as optional
    for key in _OPTIONAL_KEYS:
        assert key in why, f"optional key {key} is unnamed"


def test_a_prepared_route_emission_is_told_its_route_not_told_to_type_a_table(
        tmp_path):
    """The route-aware half.

    A wizard emission for a source the config-driven route cannot decode
    carries a ``[fetch]`` table and NO ``[case_data]`` -- by design.
    Telling that reader to hand-write ``[case_data]`` sends them to
    build an input table for a decoder that will never read it.  The
    refusal has the ``[fetch]`` table in the same dict it just searched,
    so it can say which route the file is actually on.
    """

    message = _no_table_message(
        tmp_path,
        tables='\n[fetch]\nsource = "gfs"\ncycle = "2026-07-28T06"\n')
    action, why = split(message)
    assert "gfs" in action                       # the source it read
    assert "gpuwm go" in action                  # the door for that route
    assert why
    # It must NOT steer this reader into authoring a case-data table.
    assert "gpuwm domain --source" not in action


def test_an_emission_for_the_config_driven_source_still_gets_the_table_remedy(
        tmp_path):
    """Control for the branch above: a [fetch] table naming the source the
    config-driven route DOES decode means the table was lost, not that the
    file is on another route."""

    action, _why = split(_no_table_message(
        tmp_path, tables='\n[fetch]\nsource = "era5"\ncycle = "1999-05-03T12"\n'))
    assert "gpuwm domain" in action
    assert "gpuwm go" not in action


def test_both_loaders_refuse_with_the_same_remedy(tmp_path):
    """``load_case_data`` and ``load_experiment_case`` each raise their own
    copy of this refusal.  Two copies is how one of them drifts."""

    path = _write(tmp_path)
    with pytest.raises(ValueError) as one:
        load_case_data(path)
    with pytest.raises(ValueError) as two:
        load_experiment_case(path)
    assert str(one.value) == str(two.value)


# --------------------------------------------------------------------
# A [case_data] that is present and incomplete
# --------------------------------------------------------------------

def _missing_keys_message(tmp_path) -> str:
    path = _partial_case_toml(tmp_path)
    with pytest.raises(ValueError) as caught:
        load_case_data(path)
    return str(caught.value)


def test_the_missing_key_refusal_pastes_a_line_for_each_missing_key(
        tmp_path):
    """Naming a key is not a remedy; naming the line to add is."""

    action, _why = split(_missing_keys_message(tmp_path))
    assert "missing required key" in action            # unchanged phrase
    for key in ("wps_namelist", "geog_root", "sfcp_to_sfcp"):
        assert f"{key} =" in action, f"{key} has no pasteable line"
    assert "remedy:" in action


def test_the_missing_key_remedy_does_not_offer_keys_already_declared(
        tmp_path):
    """A remedy that tells you to add what you already wrote is noise."""

    action, _why = split(_missing_keys_message(tmp_path))
    for key in ("forcing", "vtable", "output_title"):
        assert f"{key} =" not in action, f"{key} is already declared"


def test_the_missing_key_refusal_explains_what_each_key_is_for(tmp_path):
    """``--explain`` carries the mechanism: what the missing declarations
    feed, and what the table already had."""

    action, why = split(_missing_keys_message(tmp_path))
    assert why and why != action
    # A job description for each key that is actually missing.  The keys
    # the reader already declared are NOT explained -- explaining a key
    # somebody has written is the wall this layering exists to remove.
    assert "geography" in why.lower()          # geog_root's job
    assert "WPS namelist" in why               # wps_namelist's job
    assert "surface-pressure" in why           # sfcp_to_sfcp's job
    assert "Vtable" not in why
    # ... and the keys the table DID carry, so the reader sees where the
    # gap sits rather than having to diff two lists by hand.
    assert "Already declared in this table" in why
    assert "forcing" in why and "output_title" in why


# --------------------------------------------------------------------
# The front door, which is where --explain is actually typed
# --------------------------------------------------------------------

@pytest.mark.parametrize("builder", ["no_table", "missing_keys"])
def test_explain_changes_what_the_cli_prints(tmp_path, capsys, builder):
    """The defect as a user meets it: the same words twice."""

    import gpuwm.cli as cli

    path = (_write(tmp_path) if builder == "no_table"
            else _partial_case_toml(tmp_path))
    out = tmp_path / "static.nc"

    assert cli.main(["static", str(path), "--output", str(out)]) == 2
    bare = capsys.readouterr()
    assert cli.main(["static", str(path), "--output", str(out),
                     "--explain"]) == 2
    explained = capsys.readouterr()

    bare_text = bare.out + bare.err
    explained_text = explained.out + explained.err
    assert explained_text != bare_text, "--explain added nothing"
    assert len(explained_text) > len(bare_text)
    # The default layer points at the flag that produces the rest.
    assert "--explain" in bare_text


# --------------------------------------------------------------------
# No gate widening
# --------------------------------------------------------------------

def test_a_complete_case_data_table_still_loads(tmp_path):
    data = load_case_data(make_case_toml(tmp_path))
    assert data.output_title == "fixture title"
    assert data.sfcp_to_sfcp is True
