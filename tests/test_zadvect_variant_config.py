"""Numerical-generation selection must survive every forecast identity."""
from dataclasses import asdict, replace
from datetime import datetime

import pytest

from gpuwm.config import RunConfig, _validate_dynamics_coefficients


def _cfg(**kwargs):
    return RunConfig(24, 24, 8, 3000., 3000., 10000., 6., 60., **kwargs)


@pytest.mark.parametrize("variant", ["wrf_471", "wrf_legacy"])
def test_supported_generation_is_validated(variant):
    _validate_dynamics_coefficients(_cfg(zadvect_implicit=1,
                                         zadvect_implicit_variant=variant))


def test_unknown_generation_cannot_silently_change_the_operator():
    with pytest.raises(ValueError, match="split and column masses differ"):
        _validate_dynamics_coefficients(_cfg(zadvect_implicit_variant="typo"))


def test_restart_generation_flip_refuses_and_old_default_is_compatible():
    from gpuwm.io.restart import (
        _configuration_digest_values, _require_config_match,
        configuration_echo,
    )
    cfg = _cfg(zadvect_implicit=1)
    stored = asdict(cfg)
    stored.pop("zadvect_implicit_variant")
    _require_config_match(stored, cfg, "checkpoint")
    assert _configuration_digest_values(stored) == _configuration_digest_values(
        asdict(cfg))
    assert "zadvect_implicit_variant" not in configuration_echo(cfg)
    legacy = replace(cfg, zadvect_implicit_variant="wrf_legacy")
    assert configuration_echo(legacy)["zadvect_implicit_variant"] == "wrf_legacy"
    with pytest.raises(ValueError, match="zadvect_implicit_variant"):
        _require_config_match(stored, legacy, "checkpoint")
    with pytest.raises(ValueError, match="zadvect_implicit_variant"):
        _require_config_match(asdict(legacy), cfg, "checkpoint")


def test_shared_toml_selects_generation_and_binds_tree_identity():
    from gpuwm.core.model import restart_identity_payload
    from gpuwm.experiment import build_experiment
    from gpuwm.ingest.prepared_cache import (
        PREPARATION_INERT_RUN_FIELDS, compare_prepared_domain_config,
        prepared_domain_config_identity,
    )

    document = {
        "experiment": {"name": "implicit-variant",
                       "start_time": datetime(2020, 1, 1),
                       "run_seconds": 60., "restart_interval_s": 0.},
        "shared": {"nz": 8, "ztop": 10000., "zadvect_implicit": 1},
        "domain": [{"grid_id": 1, "parent_id": 0, "nx": 24, "ny": 24,
                    "i_parent_start": 1, "j_parent_start": 1,
                    "parent_grid_ratio": 1, "parent_time_step_ratio": 1,
                    "dx": 3000., "time_step": 6, "history_interval_s": 60.}],
    }
    default = build_experiment(document, source="implicit variant test")
    assert default.root.run.zadvect_implicit_variant == "wrf_471"
    assert "zadvect_implicit_variant" not in str(restart_identity_payload(default))
    document["shared"]["zadvect_implicit_variant"] = "wrf_legacy"
    legacy = build_experiment(document, source="implicit variant test")
    assert legacy.root.run.zadvect_implicit_variant == "wrf_legacy"
    assert restart_identity_payload(default) != restart_identity_payload(legacy)
    assert "run.zadvect_implicit_variant" in PREPARATION_INERT_RUN_FIELDS
    old = prepared_domain_config_identity(default.root)
    new = prepared_domain_config_identity(legacy.root)
    assert not compare_prepared_domain_config(old, new)[1]
    old["run"].pop("zadvect_implicit_variant")
    assert not compare_prepared_domain_config(old, new)[1]
    # Reuse does not forgive a different vertical grid.
    new["run"]["nz"] += 1
    assert "run.nz" in compare_prepared_domain_config(old, new)[1]


def test_scalar_transport_passes_the_numerical_generation(monkeypatch):
    from gpuwm.core import ieva, moist

    class Split(tuple):
        variant = "wrf_legacy"

    captured = []
    monkeypatch.setattr(ieva, "solve_scalar",
                        lambda *args, **kwargs: captured.append((args, kwargs)))
    moist._ieva_scalar(None, "tendency", "old scalar", Split((1, 2)),
                       "old mass", "new mass", 3.0)
    assert captured == [((None, "tendency", "old scalar", 2,
                          "old mass", "new mass", 3.0),
                         {"variant": "wrf_legacy"})]
