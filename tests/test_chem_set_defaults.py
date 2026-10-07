"""A chem set's own scheme defaults reach a bare configuration (CPU only).

The smoke set transcribes NOAA GSL's RRFS-SD smoke code, which washes smoke
out (wetdep_ls_opt = 1) and refreshes the plume every hour
(plumerisefire_frq = 60) by default; WRF-Chem's registry defaults
(wetscav_onoff = 0, plumerisefire_frq = 180) ran the smoke path with neither.
The set file's ``namelist_defaults`` fill only keys a configuration leaves
unset, on both loaders.
"""
from __future__ import annotations

import dataclasses
import tomllib
from types import MappingProxyType

import pytest

from gpuwm.config import (CHEM_RUN_FIELDS, RunConfig, apply_chem_set_defaults,
                          load_config)

_GRID = ("[grid]\nnx = 12\nny = 10\nnz = 6\ndx = 3000.0\ndy = 3000.0\n"
         "ztop = 12000.0\n[run]\ndt = 18.0\nrun_seconds = 60.0\n"
         # The smoke rows mix through vertmx, which needs a PBL's exch_h.
         "bl_pbl_physics = 1\nsf_sfclay_physics = 1\n")


def _load(tmp_path, chem: str) -> RunConfig:
    path = tmp_path / "chem.toml"
    path.write_text(_GRID + chem, encoding="utf-8")
    try:
        return load_config(path)
    except ValueError as error:
        if "unknown key" in str(error):
            pytest.skip(f"[run] does not take chem keys here: {error}")
        raise


def test_the_smoke_set_declares_its_reference_defaults():
    from gpuwm.chem_table import catalog
    row = catalog().sets["smoke"]
    assert dict(row.namelist_defaults) == {"wetscav_onoff": -1,
                                           "plumerisefire_frq": 60}
    assert set(row.namelist_defaults) <= set(CHEM_RUN_FIELDS)


def test_a_bare_smoke_configuration_runs_washout_and_an_hourly_plume(tmp_path):
    cfg = _load(tmp_path, 'chem_sets = "smoke"\nchem_sources = "rave-3km"\n')
    assert cfg.wetscav_onoff == -1
    assert cfg.plumerisefire_frq == 60
    # The RunConfig field defaults are WRF-Chem's and stay so: only the
    # set's loader default moved.
    fields = {f.name: f.default for f in dataclasses.fields(RunConfig)}
    assert fields["wetscav_onoff"] == 0 and fields["plumerisefire_frq"] == 180


def test_a_stated_key_is_never_overridden(tmp_path):
    cfg = _load(tmp_path, 'chem_sets = "smoke"\nchem_sources = "rave-3km"\n'
                          'wetscav_onoff = 0\nplumerisefire_frq = 180\n')
    assert cfg.wetscav_onoff == 0
    assert cfg.plumerisefire_frq == 180


def test_sets_without_defaults_keep_the_wrf_chem_registry_values(tmp_path):
    cfg = _load(tmp_path, 'chem_sets = "tracer_test"\n')
    assert cfg.wetscav_onoff == 0 and cfg.plumerisefire_frq == 180
    off = _load(tmp_path, "")
    assert off.chem_sets == "" and off.wetscav_onoff == 0


def _fake_sets(monkeypatch, rows: dict):
    import gpuwm.chem_table as chem_table
    real = chem_table.catalog()
    sets = dict(real.sets)
    for name, defaults in rows.items():
        sets[name] = dataclasses.replace(
            real.sets["tracer_test"], name=name,
            namelist_defaults=MappingProxyType(defaults))
    fake = dataclasses.replace(real, sets=MappingProxyType(sets))
    monkeypatch.setattr(chem_table, "catalog", lambda: fake)


def test_two_sets_defaulting_one_key_differently_are_refused(monkeypatch):
    _fake_sets(monkeypatch, {"aa": {"wetscav_onoff": -1},
                             "bb": {"wetscav_onoff": 0}})
    with pytest.raises(ValueError, match="whichever set is listed first"):
        apply_chem_set_defaults({"chem_sets": "aa,bb"})
    # Agreeing sets, or a stated key, are fine.
    values = {"chem_sets": "aa,bb", "wetscav_onoff": -1}
    assert apply_chem_set_defaults(values)["wetscav_onoff"] == -1


def test_a_set_may_not_default_a_key_outside_the_chem_block(monkeypatch):
    _fake_sets(monkeypatch, {"aa": {"time_step_sound": 6}})
    with pytest.raises(ValueError, match="retune the rest of the model"):
        apply_chem_set_defaults({"chem_sets": "aa"})


def test_the_experiment_loader_applies_the_same_defaults(tmp_path):
    from gpuwm.cli import main as cli_main
    from gpuwm.experiment import build_experiment
    out = tmp_path / "area.toml"
    assert cli_main(["domain", "--point=34.07,-118.54", "--card", "24gb",
                     "--ladder", "12", "--source", "gfs",
                     "--cycle", "2026-07-29T18", "--hours", "2",
                     "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "[shared]\n" in text
    text = text.replace("[shared]\n", '[shared]\nchem_sets = "smoke"\n'
                        'chem_sources = "rave-3km"\nchem_conv_tr = 0\n', 1)
    raw = tomllib.loads(text)
    raw.pop("fetch", None)
    experiment = build_experiment(raw, str(out))
    for domain in experiment.domains:
        assert domain.run.wetscav_onoff == -1
        assert domain.run.plumerisefire_frq == 60
