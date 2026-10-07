"""Canopy energy output and its disk inventory share one field table."""
from types import SimpleNamespace

import numpy as np

from gpuwm.config import RunConfig
from gpuwm.io.history_layout import (
    _OUTPUT_FIELDS, _SLUCM_OUTPUT_FIELDS, physics_history_fields,
    produced_history_shapes,
)
from gpuwm.io.wrf_output_schema import REGISTRY_VAR_META


def _physics(option):
    plane = np.zeros((3, 4), np.float32)
    return SimpleNamespace(fields={key: plane for key in _OUTPUT_FIELDS.values()},
        urban=None if option == 0 else SimpleNamespace(option=option,
            fields={key: plane.copy() for key in _SLUCM_OUTPUT_FIELDS.values()}),
        sase_active=False, hmix_k_diag=None, radiation_active=False,
        rainc=None, mp_physics=0, microphysics=None,
        _zero_accumulator=lambda: plane)


def test_canopy_outputs_are_live_component_buffers_and_off_keeps_inventory():
    off = physics_history_fields(_physics(0))
    urban = _physics(1)
    on = physics_history_fields(urban)
    assert set(on) - set(off) == set(_SLUCM_OUTPUT_FIELDS)
    assert set(physics_history_fields(_physics(2))) == set(off)
    for name, field in _SLUCM_OUTPUT_FIELDS.items():
        assert on[name] is urban.urban.fields[field]
        assert name in REGISTRY_VAR_META


def test_canopy_disk_inventory_prices_every_published_plane():
    from dataclasses import replace
    cfg = RunConfig(nx=12, ny=12, nz=10, dx=3000, dy=3000,
                    ztop=12000, dt=6, run_seconds=120, sf_surface_physics=2)
    off = produced_history_shapes(cfg)
    on = produced_history_shapes(replace(cfg, sf_urban_physics=1))
    assert set(on) - set(off) == set(_SLUCM_OUTPUT_FIELDS)
    assert all(on[name] == (12, 12) for name in _SLUCM_OUTPUT_FIELDS)


def test_energy_carriers_are_output_only_and_disabled_fields_stay_absent():
    from dataclasses import replace
    from gpuwm.io.restart import configuration_echo, _configuration_digest_values
    cfg = RunConfig(nx=12, ny=12, nz=10, dx=3000, dy=3000,
                    ztop=12000, dt=6, run_seconds=120, sf_surface_physics=2)
    assert "surface_energy_diag" not in configuration_echo(cfg)
    enabled = replace(cfg, surface_energy_diag=True)
    assert _configuration_digest_values(configuration_echo(cfg)) == _configuration_digest_values(configuration_echo(enabled))
    off = _physics(0)
    off.fields.update(albedo=np.full((3, 4), .2), emiss=np.full((3, 4), .95))
    before = physics_history_fields(off)
    off.surface_energy_diag = True
    after = physics_history_fields(off)
    assert set(after) - set(before) == {"ALBEDO", "EMISS"}
    assert after["ALBEDO"] is off.fields["albedo"]
    shapes_before = produced_history_shapes(cfg)
    shapes_after = produced_history_shapes(enabled)
    assert set(shapes_after) - set(shapes_before) == {"ALBEDO", "EMISS"}
    off.fields["gsw"] = np.full((3, 4), 500.0)
    assert physics_history_fields(off)["GSW"] is off.fields["gsw"]
    ruc_shapes = produced_history_shapes(replace(enabled, sf_surface_physics=3,
                                                 num_soil_layers=6))
    assert ruc_shapes["GSW"] == (12, 12)


def test_default_public_and_prepared_domain_documents_omit_new_output_switch():
    from dataclasses import replace
    from gpuwm.experiment import DomainConfig, domain_config_document
    from gpuwm.ingest.prepared_cache import (
        prepared_domain_config_identity, compare_prepared_domain_config,
        INERT_DIAGNOSTIC_IDENTITY_FIELDS,
    )
    cfg = RunConfig(nx=12, ny=12, nz=10, dx=3000, dy=3000,
                    ztop=12000, dt=6, run_seconds=120, sf_surface_physics=2)
    domain = DomainConfig(grid_id=1, parent_id=0, i_parent_start=1,
        j_parent_start=1, parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=3600, run=cfg)
    before = domain_config_document(domain)
    assert "surface_energy_diag" not in before["run"]
    assert prepared_domain_config_identity(domain) == before
    after = domain_config_document(replace(domain,
        run=replace(cfg, surface_energy_diag=True)))
    assert after["run"]["surface_energy_diag"] is True
    assert "run.surface_energy_diag" in INERT_DIAGNOSTIC_IDENTITY_FIELDS
    assert compare_prepared_domain_config(before, after) == ([], [])
