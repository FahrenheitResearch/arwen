"""Opt-in chemistry preserves the public staging configuration bytes."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm.experiment import (build_experiment, domain_config_document,
                              experiment_config_document)

BASELINE_SHA = "deedb07db775bf1455dc1a353bcf3d9b75d00296"
FIXTURE = Path(__file__).parent / "data/aq_off_emitted_experiment_bytes.json"


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str).encode()


def _assert_neutral_cycling_omitted(document):
    """Keep the independent emitted bytes under neutral cycling.

    c99b549e6 restores omission of the inert False default from the public
    document, matching restart identity. Its resolved config value remains
    False; a selected True value must remain explicit below.
    """
    import copy
    historical = copy.deepcopy(document)
    domains = historical.get("domains", (historical,))
    for domain in domains:
        assert "cycling" not in domain["run"]
    return historical


@pytest.mark.parametrize("case", ["generic_dry", "non_feature_physical"])
@pytest.mark.parametrize("inactive_nondefaults", [False, True])
def test_public_emitted_config_bytes_match_independent_staging(
        case, inactive_nondefaults):
    baseline = json.loads(FIXTURE.read_text())
    assert baseline["baseline_sha"] == BASELINE_SHA
    row = baseline["cases"][case]
    assert hashlib.sha256(row["input_toml"].encode()).hexdigest() == row["input_sha256"]
    raw = tomllib.loads(row["input_toml"])
    raw.pop("fetch", None)
    exp = build_experiment(raw, source=baseline["source_label"])
    if inactive_nondefaults:
        exp = replace(exp, domains=tuple(replace(
            domain, run=replace(domain.run, chem_adv_opt=2, kemit=17,
                                dust_alpha=3.0, aer_ra_feedback=1))
            for domain in exp.domains))
    assert all(domain.run.cycling is False for domain in exp.domains)
    selected = replace(exp, domains=tuple(replace(
        domain, run=replace(domain.run, cycling=True)) for domain in exp.domains))
    assert all(domain["run"]["cycling"] is True
               for domain in experiment_config_document(selected)["domains"])
    actual = _encode(_assert_neutral_cycling_omitted(experiment_config_document(exp)))
    # The independently captured current baseline includes MYNN's three
    # public defaults. The named input also explicitly selects solar albedo;
    # neither declared difference belongs to the chemistry-off block.
    expected_run = json.loads(row["domain_json"][0])["run"]
    assert "cycling" not in expected_run
    assert {name: expected_run[name] for name in (
        "bl_mynn_version", "bl_mynn_gsd41_unsquared_qtke",
        "bl_mynn_cloud_tendency_form")} == {
            "bl_mynn_version": "wrf_461",
            "bl_mynn_gsd41_unsquared_qtke": False,
            "bl_mynn_cloud_tendency_form": "wrf_461"}
    assert expected_run.get("alb_sol", 0) == (case == "non_feature_physical")
    # Current staging omits these neutral public values. The resolved
    # selections remain explicit defaults, rather than disappearing from
    # the forecast contract or being changed to a different generation.
    neutral = {"thompson_version": "wrf_461", "thompson_fork_snow_fall": "blend",
               "rrtmg_cloud_optics_form": "wrf_461", "rrtmg_smoke_manifest": ""}
    for name, default in neutral.items():
        assert name not in expected_run
        assert all(getattr(domain.run, name) == default for domain in exp.domains)
    if "terrain_clock" not in row["input_toml"]:
        # Declared default correction, 2.8.8: an input that leaves
        # terrain_clock unset reads "local_face" (gpuwm/terrain_clock_local.py);
        # the baseline was captured under "measured".  The default moves
        # that one value and nothing else; the pinned bytes below are read
        # under the clock they were captured with.
        expected = row["experiment_json"]
        assert expected.count('"terrain_clock":"measured"') == 1
        assert actual == expected.replace(
            '"terrain_clock":"measured"',
            '"terrain_clock":"local_face"').encode()
        exp = replace(exp, domains=tuple(replace(
            domain, run=replace(domain.run, terrain_clock="measured"))
            for domain in exp.domains))
        actual = _encode(_assert_neutral_cycling_omitted(experiment_config_document(exp)))
    assert actual == row["experiment_json"].encode()
    assert hashlib.sha256(actual).hexdigest() == row["experiment_sha256"]
    for domain, expected, digest in zip(
            exp.domains, row["domain_json"], row["domain_sha256"]):
        actual_domain = _encode(_assert_neutral_cycling_omitted(domain_config_document(domain)))
        assert actual_domain == expected.encode()
        assert hashlib.sha256(actual_domain).hexdigest() == digest
