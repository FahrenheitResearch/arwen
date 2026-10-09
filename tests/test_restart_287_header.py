"""A checkpoint 2.8.7 wrote resumes under 2.8.8 when the user does what the
refusal says, and a changed scheme is refused first, by name.

The breakage these guard, measured on the 2.8.8 candidate with the real
2.8.7 headers recorded under tests/data/restart_287_headers/:

* terrain-clock resume (``sfire-ideal-4s.json``, the shipped
  examples/sfire-ideal run under the released 2.8.7 wheel, its 4 s
  checkpoint; every scheme disabled, so no identity moved in 2.8.8).
  Resumed under 2.8.8's default the configuration walk refuses with the
  note to write ``terrain_clock = "measured"``; written, the physics gate
  rebuilt the header's configuration digest under RunConfig's NEW default
  ("local_face") and refused again on ``configuration_sha256``.  A header
  without the key ran "measured", so the rebuild must read it so.
* changed-scheme resume: a checkpoint of a scheme 2.8.8 computes
  differently resumes under no configuration, yet the walk sent the user
  to the terrain-clock note first and the gate then named only
  "algorithms, configuration_sha256".  The pre-check names the scheme.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from gpuwm.config import TERRAIN_CLOCK_RESTART_BREAK_NOTICE, RunConfig
from gpuwm.io import restart
from test_restart import _fill_setup, _shim_state

DATA = Path(__file__).parent / "data" / "restart_287_headers"


def _header(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _run_config(config, **changes):
    names = {field.name for field in dataclasses.fields(RunConfig)}
    values = {key: value for key, value in config.items() if key in names}
    values.update(changes)
    return RunConfig(**values)


def test_the_recorded_header_is_a_released_287_one_without_the_clock_key():
    header = _header("sfire-ideal-4s.json")
    assert header["producer"]["version"] == "2.8.7"
    assert "terrain_clock" not in header["config"]
    assert restart._json_sha256(header["physics_setup"]) == \
        header["physics_setup_fingerprint"]


def test_the_stored_digest_rebuilds_under_the_clock_the_header_ran():
    header = _header("sfire-ideal-4s.json")
    stored = header["physics_setup"]["configuration_sha256"]
    rebuilt = restart._run_config_from_header(header["config"])
    assert rebuilt.terrain_clock == "measured"
    assert restart._configuration_fingerprint(rebuilt) == stored
    # The defect: the dataclass default is 2.8.8's "local_face", and a
    # rebuild that takes it can never reproduce a 2.8.7 digest.
    defaulted = _run_config(header["config"])
    assert defaulted.terrain_clock == "local_face"
    assert restart._configuration_fingerprint(defaulted) != stored


def test_a_287_checkpoint_of_unchanged_schemes_resumes_under_measured(
        monkeypatch):
    header = _header("sfire-ideal-4s.json")
    live = _run_config(header["config"], terrain_clock="measured")
    restart._require_current_algorithm_identities(header, live, "recorded")
    restart._require_config_match(header["config"], live, "recorded")
    # The live identity is the recorded one with THIS build's digest of the
    # live configuration: the resolved driver, tables and assets of the run
    # that wrote it (which a CPU shim cannot rebuild) are held equal, and
    # the gate's rebuild of the stored digest is what is asked.
    recorded = header["physics_setup"]
    monkeypatch.setattr(
        restart, "physics_setup_identity",
        lambda state, cfg: dict(
            recorded,
            configuration_sha256=restart._configuration_fingerprint(cfg)))
    state = _shim_state(live, monkeypatch)
    _fill_setup(state)
    restart._require_physics_setup_match(header, state, live, "recorded")


def test_the_default_clock_is_refused_with_the_note_that_resumes_it():
    header = _header("sfire-ideal-4s.json")
    live = _run_config(header["config"])
    restart._require_current_algorithm_identities(header, live, "recorded")
    with pytest.raises(restart.RestartMismatchError,
                       match="terrain_clock") as caught:
        restart._require_config_match(header["config"], live, "recorded")
    assert TERRAIN_CLOCK_RESTART_BREAK_NOTICE in str(caught.value)


def test_a_changed_scheme_is_refused_first_and_named(monkeypatch):
    """The recorded header, its microphysics identity set to 2.8.7's
    Kessler string as a run of mp_physics = 1 under 2.8.7 wrote it: the
    refusal comes before the configuration walk (so before its
    terrain-clock note), names the selector and both identities, and says
    no configuration change resumes it."""
    header = _header("sfire-ideal-4s.json")
    header["config"] = dict(header["config"], mp_physics=1)
    setup = dict(header["physics_setup"])
    setup["algorithms"] = dict(setup["algorithms"],
                               microphysics="kessler-warm-rain-v1")
    header["physics_setup"] = setup
    header["physics_setup_fingerprint"] = restart._json_sha256(setup)
    live = _run_config(header["config"])
    with pytest.raises(restart.RestartMismatchError) as caught:
        restart._require_current_algorithm_identities(
            header, live, "recorded")
    message = str(caught.value)
    assert ("mp_physics=1 (microphysics): kessler-warm-rain-v1 -> "
            + restart.MICROPHYSICS_ALGORITHM_IDENTITIES[1]) in message
    assert "no configuration change resumes it" in message
    assert "terrain_clock" not in message


def test_a_tampered_identity_is_left_to_the_integrity_refusal():
    """An identity that no longer matches its own fingerprint is a damaged
    file, not a changed build; the pre-check must not reword it."""
    header = _header("sfire-ideal-4s.json")
    header["physics_setup"] = dict(
        header["physics_setup"],
        algorithms=dict(header["physics_setup"]["algorithms"],
                        microphysics="silently-tampered"))
    live = _run_config(header["config"], terrain_clock="measured")
    restart._require_current_algorithm_identities(header, live, "recorded")
