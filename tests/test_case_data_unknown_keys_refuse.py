"""A ``[case_data]`` key this schema does not know is a wrong answer.

This is the third appearance of one defect.  ``gpuwm/experiment.py``'s
``_reject_unknown_keys`` warned and dropped (fixed in ``a4417efb``);
``twentycrv3_wrf.main``'s translation of the ``[case_data]`` refusal
stopped firing when unknown TABLES were switched to warning (restored in
``83720e6c``); and ``build_case_data`` -- in a module whose own docstring
says "fail-loud" -- warned and dropped unknown ``[case_data]`` keys.

Exposure is the five OPTIONAL keys, because those are optional by
construction: nothing after the required-key check ever looks for them,
so a misspelling of any one passes every remaining check and the case
loads with the built-in default under the name of the user's value::

    forcing_interval_s  output_domain  source_orography
    source_orography_variable  co2_vmr

``forcing_interval_s`` is the one that reaches the numbers, in two
places.  It is enforced against the decoded schedule
(:func:`gpuwm.runtime.forcing_schedule`), and -- the part that changes a
forecast rather than merely failing to guard one -- it SELECTS the
forcing window the input catalog consumes: ``_select_contiguous_times``
keeps the unique longest contiguous run at the declared interval and
excludes everything else.  A CDS retrieval that returns a date x time
cross-product has non-contiguous extras in it; the declared interval is
what removes them.

Measured on this tree against the real May-1999 ERA5 case, through the
shipped CLI, before the fix:

  ``forcing_interval_s   = 21600.0``  ->  rc 0, PASS, 3 valid times,
                                          43200 s of coverage
  ``forcing_interval_s   = 10800.0``  ->  rc 2, "declared
                                          forcing_interval_s = 10800 but
                                          the decoded schedule steps ...
                                          (21600 s)", no wrfout
  ``forcing_interval_sec = 10800.0``  ->  rc 0, ONE stderr warning,
                                          ``time.forcing_interval_s =
                                          discover``, 5 valid times,
                                          108000 s of coverage, two
                                          wrfout frames, ``nan_free
                                          True``

The same number, one letter apart, is either a refusal or a completed
forecast built on a forcing window nobody selected.

Every test here fails against the tree as it stood at ``b654a3e0``.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from gpuwm.case_data import _KNOWN_KEYS, _OPTIONAL_KEYS, load_case_data
from gpuwm.explain import split

from test_case_data import _CASE_DATA_TOML, make_case_toml


# --------------------------------------------------------------------
# The five optional keys, and why the checks below them cannot help
# --------------------------------------------------------------------

def test_the_optional_keys_are_exactly_the_silent_exposure():
    """Pin the enumeration this defect's blast radius is measured on.

    A key added to ``_OPTIONAL_KEYS`` without a thought about
    misspelling is a new silent drop; this fails when the set moves.
    """

    assert set(_OPTIONAL_KEYS) == {
        "forcing_interval_s", "output_domain", "source_orography",
        "source_orography_variable", "co2_vmr",
    }


@pytest.mark.parametrize("key", sorted(_OPTIONAL_KEYS))
def test_a_misspelled_optional_key_refuses_and_names_both(tmp_path, key):
    """One letter added to each optional key.  Every one used to load."""

    typo = key + "x"
    path = make_case_toml(
        tmp_path, case_data=_CASE_DATA_TOML + f"{typo} = 1\n")
    with pytest.raises(ValueError) as caught:
        load_case_data(path)
    message = str(caught.value)
    assert typo in message               # names the key it refused
    assert "[case_data]" in message      # names the table
    assert key in message                # names the key it thinks was meant


def test_the_headline_misspelling_refuses(tmp_path):
    """``forcing_interval_sec`` -- the exhibit from the module docstring."""

    path = make_case_toml(
        tmp_path,
        case_data=_CASE_DATA_TOML + "forcing_interval_sec = 10800.0\n")
    with pytest.raises(ValueError) as caught:
        load_case_data(path)
    message = str(caught.value)
    assert "forcing_interval_sec" in message
    assert "did you mean 'forcing_interval_s'?" in message


def test_the_refusal_is_one_sentence_with_the_key_list_behind_explain(
        tmp_path):
    """Message-layer contract, same as the [shared]/[[domain]] refusal:
    the known-key list is mechanism and belongs behind ``--explain``."""

    path = make_case_toml(
        tmp_path, case_data=_CASE_DATA_TOML + "mystery_key = 1\n")
    with pytest.raises(ValueError) as caught:
        load_case_data(path)
    action, why = split(str(caught.value))
    assert why, "the known-key list belongs behind --explain"
    assert "Known [case_data] keys" in why
    assert action.count("\n") <= 2, action
    # The action half must not carry the list it was layered away from.
    assert "output_domain" not in action


# --------------------------------------------------------------------
# The exit code, not just the exception
# --------------------------------------------------------------------

def test_an_unknown_case_data_key_refuses_through_the_cli_at_rc_2(tmp_path):
    """A test that only asserts "a warning was emitted" is what let the
    previous instance get relaxed.  Assert the refusal AND the code the
    shipped entry point exits with."""

    import gpuwm.cli as cli

    path = make_case_toml(
        tmp_path,
        case_data=_CASE_DATA_TOML + "forcing_interval_sec = 10800.0\n")
    assert cli.main(["check", str(path), "--budget-gib", "2",
                     "--vram-gib", "6"]) == 2


def test_a_valid_case_data_table_still_loads(tmp_path):
    """No gate widening and no gate narrowing: the fixture table -- which
    declares all five optional keys spelled correctly -- is unaffected."""

    data = load_case_data(make_case_toml(tmp_path))
    assert data.forcing_interval_s == 21600.0
    assert data.output_domain == 1
    assert data.co2_vmr == 330.0e-6
    assert set(_OPTIONAL_KEYS) <= _KNOWN_KEYS


# --------------------------------------------------------------------
# Why the drop is a wrong forecast and not merely a missing guard
# --------------------------------------------------------------------

def test_the_declared_interval_selects_the_forcing_window():
    """The numeric consequence of dropping ``forcing_interval_s``.

    These are the six valid times the real May-1999 ERA5 pair decodes to
    (a CDS date x time cross-product).  Declared, the interval keeps the
    3-snapshot contiguous run the run actually wants; dropped, all six
    reach the catalog and the run ceiling triples.
    """

    from gpuwm.ingest.preflight import _select_contiguous_times

    raw = (datetime(1999, 5, 3, 0), datetime(1999, 5, 3, 12),
           datetime(1999, 5, 3, 18), datetime(1999, 5, 4, 0),
           datetime(1999, 5, 4, 12), datetime(1999, 5, 4, 18))

    declared, excluded, _why = _select_contiguous_times(raw, 21600.0)
    assert declared == (datetime(1999, 5, 3, 12), datetime(1999, 5, 3, 18),
                        datetime(1999, 5, 4, 0))
    assert len(excluded) == 3
    # The run ceiling `gpuwm check` printed for the correctly spelled
    # config.
    assert (declared[-1] - declared[0]).total_seconds() == 43200.0

    # What the misspelling produced: the key never arrives, so nothing
    # is excluded and the forecast is built on a different window --
    # the 151200 s ceiling `gpuwm check` printed while reporting PASS.
    dropped, dropped_excluded, _why = _select_contiguous_times(raw, None)
    assert dropped == raw
    assert dropped_excluded == ()
    assert (dropped[-1] - dropped[0]).total_seconds() == 151200.0


# --------------------------------------------------------------------
# The per-domain half, and the documentation that denied it existed
# --------------------------------------------------------------------

@pytest.mark.parametrize("key,value", [
    ("bl_pbl_physics", 5),   # a per-nest PBL scheme WRF does carry
    ("hill_height", 100.0),  # a live terrain setting
    ("isfflx", 1),           # nentries=1 in WRF; no per-domain spelling
])
def test_an_unsupported_per_domain_key_refuses(tmp_path, key, value):
    """The sibling defect: a `[[domain]]` key this loader does not read.

    Dropped, it ran the `[shared]` scheme on every nest and reported
    success, with nothing in the receipt contradicting what the user
    wrote.  ``a4417efb`` closed this by making the shared
    ``_reject_unknown_keys`` refuse; the bar is here so it stays closed
    on the release line, since the branch that fixed it independently
    is held for a later version.
    """

    from gpuwm.experiment import load_experiment

    from test_experiment import _write

    with pytest.raises(ValueError) as caught:
        load_experiment(_write(tmp_path, d02=f"{key} = {value!r}"))
    message = str(caught.value)
    assert key in message
    assert "does not have a key" in message


def test_the_per_domain_override_list_is_documented(tmp_path):
    """An override reachable only by hand-written TOML, whose reference
    says the opposite by omission, is the same defect from the other
    side: the user cannot tell what their config will do.

    ``docs/public/CONFIGURATION.md`` opens the dynamics table with
    "shared across domains unless marked per-domain", and only three of
    the twelve rows carry that mark.  The list is now stated once; this
    keeps it in step with the code.
    """

    from pathlib import Path

    from gpuwm.experiment import _DOMAIN_RUN_OVERRIDES

    doc = (Path(__file__).resolve().parents[1]
           / "docs" / "public" / "CONFIGURATION.md").read_text(
               encoding="utf-8")
    _head, _mark, tail = doc.partition(
        "**Which keys a `[[domain]]` table may override.**")
    assert _mark, "the per-domain override list is missing from the doc"
    block = tail.split("Only `gpuwm domain`")[0]
    for key in _DOMAIN_RUN_OVERRIDES:
        assert key in block, f"{key} is per-domain in code, absent from doc"
    assert f"these {len(_DOMAIN_RUN_OVERRIDES)}" in tail.lower() or (
        "twelve" in tail and len(_DOMAIN_RUN_OVERRIDES) == 12)


# --------------------------------------------------------------------
# The same defect one module over: gpuwm/adapt.py's descriptor loader
# --------------------------------------------------------------------

def test_an_unknown_adapt_descriptor_key_refuses(tmp_path):
    """``adapt._object`` carried the identical falsified sentence -- "the
    required-key check below still catches every typo of a key that
    matters".  ``soil_policy.target_layers_m`` is the counterexample: it
    is allowed-but-not-required, so under ``identity_complete_layers`` a
    misspelling of it warned once and adapted the source inventory
    unchanged."""

    from gpuwm.adapt import _load_adapt_policy

    from test_adapt import _fill_scaffold, _skeleton_payload

    _path, payload = _skeleton_payload(tmp_path)
    filled = _fill_scaffold(payload)
    filled["adapt"]["soil_policy"]["target_layers_metres"] = [[0.0, 0.1]]
    descriptor = tmp_path / "typo.descriptor.json"
    descriptor.write_text(json.dumps(filled, indent=2), encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        _load_adapt_policy(descriptor)
    message = str(caught.value)
    assert "target_layers_metres" in message
    assert "descriptor.adapt.soil_policy" in message
    assert "did you mean 'target_layers_m'?" in message


def test_a_valid_adapt_descriptor_still_loads(tmp_path):
    """Control: the descriptor the skeleton round-trip already pins."""

    from gpuwm.adapt import _load_adapt_policy

    from test_adapt import _fill_scaffold, _skeleton_payload

    _path, payload = _skeleton_payload(tmp_path)
    descriptor = tmp_path / "filled.descriptor.json"
    descriptor.write_text(json.dumps(_fill_scaffold(payload), indent=2),
                          encoding="utf-8")
    policy, _snapshot = _load_adapt_policy(descriptor)
    assert policy["model_top_pa"] == 5000.0
