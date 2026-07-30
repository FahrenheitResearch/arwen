"""CPU-only contract gates for the combined rented-GPU smoke."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from gpuwm.config import radiation_scheme_ids


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "smoke_easy_physics_gpu.py"


def _module():
    spec = importlib.util.spec_from_file_location("easy_physics_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_combined_smoke_config_uses_executable_pair_and_every_step_cadence():
    mod = _module()
    cfg = mod._combined_config(0)
    assert cfg.moist and cfg.mp_physics == 6 and cfg.wsm6_hail_opt == 0
    assert cfg.ra_physics == 0
    assert radiation_scheme_ids(cfg) == (0, 1)
    assert cfg.radt == 0.0 and cfg.radt_minutes == 0.0


def test_combined_smoke_gate_requires_both_scheduled_schemes():
    mod = _module()
    good = dict(
        state_finite=True, moisture_min=0.0, condensate_max=1.0e-4,
        moisture_change_max=1.0e-5, wsm6_heating_max=1.0e-4,
        sw_heating_max=1.0e-5, sw_heating_min=0.0,
        swdown_min=100.0, swdown_max=500.0,
        lw_heating_max=0.0, glw_error=0.0,
        precip_min=0.0, sr_min=0.0, sr_max=1.0, elapsed_error=0.0,
        radiation_calls=4, adapter_updates=4, microphysics_updates=4,
        steps=4, scheme_ids=(0, 1))
    assert mod._failure_reasons(**good) == []
    bad = dict(good, radiation_calls=0, microphysics_updates=0,
               scheme_ids=(1, 1))
    failures = mod._failure_reasons(**bad)
    assert any("scheduler" in value for value in failures)
    assert any("WSM6 diagnostics" in value for value in failures)
    assert any("LW/SW=0/1" in value for value in failures)
