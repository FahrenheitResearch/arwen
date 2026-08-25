"""The registry's DISPLAY-NAME and CREDENTIAL columns.

Two facts a front end used to have to invent for itself:

* the human NAME of a source.  Every consumer showed the raw registry id
  (``gem-gdps``, ``aigefs``, ``hrrr-prs``) because the rows carried no
  name, and any front end that wanted one had to keep a per-model lookup
  -- exactly the table the arbitrary acceptance test forbids.
* what a source needs CONFIGURED before its bytes can be acquired.  The
  rows recorded nothing, so a GUI carried a one-row credential table as
  a hardcoded exception.

Both are columns of the row now, so adding a model stays table work: the
name and the credential facts travel with the row into
``run-plan --sources`` with zero code change on either side of the seam.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import re

import pytest

from gpuwm import source_adapters as registry
from gpuwm.source_credentials import (
    CredentialLocation,
    SourceCredential,
    credential_facts,
    credential_location_display,
    credential_present,
    credentials_block,
)


def _fake_credential(**overrides) -> SourceCredential:
    facts = {
        "credential_id": "probe-arbitrary-credential",
        "display_name": "Probe Arbitrary API key",
        "location_kind": CredentialLocation.ENV_VAR,
        "location": "GPUWM_PROBE_ARBITRARY_KEY",
        "needed_for": "acquisition",
        "breakage": (
            "the acquisition step cannot authenticate and the download "
            "fails with the provider's own rejection"
        ),
        "obtain_url": "https://example.invalid/keys",
    }
    facts.update(overrides)
    return SourceCredential(**facts)


# ---------------------------------------------------------------------------
# (a) display name
# ---------------------------------------------------------------------------

def test_every_registry_row_declares_a_human_display_name():
    """A name per row, in the table -- not a lookup in a front end."""

    nameless = [adapter.source_id for adapter in registry.source_adapters()
                if not (adapter.display_name or "").strip()]
    assert nameless == []
    raw = [adapter.source_id for adapter in registry.source_adapters()
           if adapter.display_name == adapter.source_id]
    assert raw == []


def test_display_names_are_unique_across_the_registry():
    names = [adapter.display_name for adapter in registry.source_adapters()]
    assert len(set(names)) == len(names)


def test_a_row_that_omits_a_display_name_reads_back_as_its_id():
    """The fallback is the id, never ``None`` and never a traceback.

    A row added by someone who did not fill the column still renders.
    """

    grafted = dataclasses.replace(
        registry.source_adapters()[0], source_id="probe-arbitrary-model",
        display_name=None)
    assert grafted.display_title == "probe-arbitrary-model"


# ---------------------------------------------------------------------------
# (b) credentials
# ---------------------------------------------------------------------------

def test_every_row_declares_a_credential_tuple():
    for adapter in registry.source_adapters():
        assert isinstance(adapter.credentials, tuple)
        for credential in adapter.credentials:
            assert isinstance(credential, SourceCredential)


def test_the_reanalysis_row_declares_its_key_where_the_fetch_code_reads_it():
    """The declared location is the file the ERA5 route actually reads.

    Two spellings of one fact would drift; this cell fails the moment
    the registry's declared location stops naming the file
    ``gpuwm.fetch`` stats.
    """

    from gpuwm import fetch

    adapter = registry.get_source_adapter("era5")
    assert len(adapter.credentials) == 1
    credential = adapter.credentials[0]
    assert credential.location_kind is CredentialLocation.HOME_FILE
    assert credential.location == fetch.CDSAPIRC_NAME
    assert credential_location_display(credential) == str(
        fetch.cds_credentials_path())
    assert credential_present(credential) is fetch.cds_credentials_present()


def test_a_row_with_no_credential_says_that_and_claims_nothing_more():
    block = credentials_block(())
    assert block["required"] is False
    assert block["items"] == []
    # NOT "this source is public": the table records an absence of a
    # declaration, and a front end must not upgrade that to a promise.
    assert "declares no credential" in block["summary"]


def test_a_home_file_credential_reads_its_presence_from_the_declared_path(
        tmp_path, monkeypatch):
    credential = _fake_credential(
        location_kind=CredentialLocation.HOME_FILE, location=".probe-key")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert credential_present(credential) is False
    facts = credential_facts(credential)
    assert facts["present"] is False
    assert facts["status_message"] == facts["absent_message"]
    assert str(tmp_path / ".probe-key") in facts["absent_message"]

    (tmp_path / ".probe-key").write_text("key: x\n", encoding="utf-8")
    assert credential_present(credential) is True
    facts = credential_facts(credential)
    assert facts["present"] is True
    assert facts["status_message"] == facts["present_message"]


def test_an_env_var_credential_reads_presence_and_never_echoes_the_value(
        monkeypatch):
    credential = _fake_credential()
    monkeypatch.delenv(credential.location, raising=False)
    assert credential_present(credential) is False

    monkeypatch.setenv(credential.location, "s3cr3t-probe-value")
    facts = credential_facts(credential)
    assert facts["present"] is True
    assert "s3cr3t-probe-value" not in json.dumps(facts)
    assert credential.location in facts["location_display"]


def test_the_facts_are_the_declared_strings_and_the_breakage_is_named():
    credential = _fake_credential()
    facts = credential_facts(credential)
    assert facts["credential_id"] == credential.credential_id
    assert facts["display_name"] == credential.display_name
    assert facts["location_kind"] == CredentialLocation.ENV_VAR.value
    assert facts["location"] == credential.location
    assert facts["needed_for"] == credential.needed_for
    assert facts["breakage"] == credential.breakage
    assert facts["obtain_url"] == credential.obtain_url
    # The remedy travels with the refusal: a reader is told what breaks
    # and where to get the thing that fixes it.
    assert credential.breakage in facts["absent_message"]
    assert credential.obtain_url in facts["absent_message"]


def test_a_declared_credential_needs_its_facts():
    with pytest.raises(ValueError):
        _fake_credential(display_name="  ")
    with pytest.raises(ValueError):
        _fake_credential(location="")
    with pytest.raises(ValueError):
        _fake_credential(breakage="")
    with pytest.raises(TypeError):
        _fake_credential(location_kind="home_file")


def test_the_terminal_note_is_one_line_per_credential(monkeypatch, tmp_path):
    """The wizard has a measured line budget, and this must fit in it.

    MEASURED breakage: a wrapped six-line paragraph here pushed the
    wizard's default output from 14 lines to 19 and broke
    ``test_wizard_output_is_layered_and_the_default_fits_a_screen`` --
    the numbered block a reader is meant to act on stops being one when
    it scrolls.  One printed line per absent credential, so a path or a
    URL also survives intact for copying.
    """

    from gpuwm.source_credentials import absent_credential_notes

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    credential = _fake_credential(
        location_kind=CredentialLocation.HOME_FILE, location=".probe-key")
    notes = absent_credential_notes((credential, _fake_credential()))
    assert len(notes) == 2
    assert all("\n" not in note for note in notes)
    assert str(tmp_path / ".probe-key") in notes[0]
    assert credential.obtain_url in notes[0]

    (tmp_path / ".probe-key").write_text("key: x\n", encoding="utf-8")
    monkeypatch.setenv("GPUWM_PROBE_ARBITRARY_KEY", "s3cr3t")
    assert absent_credential_notes((credential, _fake_credential())) == []


def test_the_credential_derivation_names_no_source():
    """THE arbitrary gate on this module: no model, anywhere in it.

    A per-model arm here would be the hardcoded table moved one repo
    left.  Every registered id, matched whole-word and case-blind,
    must be absent from the derivation's own text.
    """

    text = Path(registry.__file__).with_name("source_credentials.py").read_text(
        encoding="utf-8")
    named = sorted({
        adapter.source_id for adapter in registry.source_adapters()
        if re.search(rf"\b{re.escape(adapter.source_id)}\b", text,
                     flags=re.IGNORECASE)})
    assert named == []


def test_a_grafted_row_carries_both_columns_through_the_manifest():
    """Adding a source is a row: the manifest repeats what the row says."""

    grafted = dataclasses.replace(
        registry.source_adapters()[0],
        source_id="probe-arbitrary-model",
        display_name="Probe Arbitrary Model",
        credentials=(_fake_credential(),))
    entry = grafted.to_dict()
    assert entry["display_name"] == "Probe Arbitrary Model"
    assert entry["credentials"][0]["credential_id"] == (
        "probe-arbitrary-credential")
    assert entry["credentials"][0]["location_kind"] == "env_var"
    # The manifest is a JSON document; a dataclass or an Enum left in it
    # would break every consumer that serialises it.
    json.dumps(entry)


def test_the_capability_manifest_stays_json():
    json.dumps(registry.source_capability_manifest())


# ---------------------------------------------------------------------------
# The CYCLE-GRID column: when a source initializes, and when its bytes land
# ---------------------------------------------------------------------------


def test_a_declared_grid_agrees_with_the_route_table_it_repeats():
    """One schedule, two places it may be written, never two answers.

    A row may declare a grid the fetch route also declares (the legacy
    transports do not, today, but nothing stops a future row from
    doing both).  Where both exist they must agree, or `--cycle latest`
    and `gpuwm fetch` would disagree about which cycles this producer
    runs -- silently, and only for that one source.
    """

    from gpuwm import fetch_routes
    from gpuwm.source_cycles import route_cycle_grid

    for adapter in registry.source_adapters():
        if adapter.cycle_grid is None:
            continue
        derived = route_cycle_grid(adapter.source_id)
        if derived is None:
            continue
        assert adapter.cycle_grid.hours == derived.hours, (
            f"{adapter.source_id}: the row says {adapter.cycle_grid.hours} "
            f"and {fetch_routes.ROUTE_TABLE_NAME} says {derived.hours}")


def test_the_hourly_models_horizon_rule_matches_the_module_that_owns_it():
    """The declared horizon is a REPEAT, and a repeat needs a guard.

    The row states which cycle hours run long, in the same shape the
    route table states it.  ``gpuwm.hrrr_forecast`` is where that rule
    is implemented; if the two drift, `--cycle latest` starts offering a
    cycle whose ladder cannot reach the requested window.
    """

    from gpuwm.hrrr_forecast import (HRRR_EXTENDED_CYCLE_HOURS,
                                     HRRR_EXTENDED_HORIZON_HOURS,
                                     HRRR_STANDARD_HORIZON_HOURS)

    grid = registry.get_source_adapter("hrrr").cycle_grid
    assert grid is not None
    assert grid.horizons == (
        (tuple(sorted(HRRR_EXTENDED_CYCLE_HOURS)),
         HRRR_EXTENDED_HORIZON_HOURS),
        (None, HRRR_STANDARD_HORIZON_HOURS))


def test_a_grid_declaration_states_its_basis():
    """A number nobody can trace is a number nobody can correct."""

    for adapter in registry.source_adapters():
        if adapter.cycle_grid is None:
            continue
        assert adapter.cycle_grid.basis.strip(), (
            f"{adapter.source_id} declares a cycle grid with no basis")


def test_the_grid_column_travels_through_the_manifest_as_json():
    """Adding a source is a row: the manifest repeats what the row says."""

    from gpuwm.source_cycles import CycleGrid

    grafted = dataclasses.replace(
        registry.source_adapters()[0],
        source_id="probe-arbitrary-model",
        cycle_grid=CycleGrid(hours=(0, 12), delay_hours=2.5,
                             search_hours=36, basis="probe"))
    entry = grafted.to_dict()
    assert entry["cycle_grid"] == {
        "hours": [0, 12], "delay_hours": 2.5, "search_hours": 36,
        "basis": "probe", "record_end": None, "horizons": []}
    json.dumps(entry)


def test_a_grid_refuses_a_shape_that_would_resolve_to_nothing():
    """The declaration is checked where it is written, not at use."""

    from gpuwm.source_cycles import CycleGrid

    with pytest.raises(ValueError, match="at least one UTC hour"):
        CycleGrid(hours=())
    with pytest.raises(ValueError, match="sorted and unique"):
        CycleGrid(hours=(12, 0))
    with pytest.raises(ValueError, match="UTC hours of day"):
        CycleGrid(hours=(0, 24))
    with pytest.raises(ValueError, match="positive window"):
        CycleGrid(hours=(0,), search_hours=0)
