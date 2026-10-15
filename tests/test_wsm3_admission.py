"""WSM3 (mp_physics = 3) is reachable from a bare namelist, default-on.

THE BREAKAGE THIS PREVENTS
--------------------------
Before the WSM3 port every one of these doors refused or silently mishandled
mp_physics = 3: the importer refused ``mp_physics = 3: no ratified gpuwm
mapping`` (21 namelists in the WRF census, the WRF 4.6.1 ndown_1 test case
among them), the config loader refused the value, the dispatcher raised
``unknown mp_physics=3``, and the wrfinput door listed WSM3 as a package it
does not carry.  A scheme that runs but cannot be reached is not shipped.

It also pins the WRF facts the rest of the model keys on for mp=3: the
Registry package is moist:qv,qc,qr with the three radii, so F_QI, F_QS and
F_QG are false; module_physics_init.F leaves warm_rain false; RRTMG gets the
scheme's radii; Kain-Fritsch takes its no-snow feedback; a mixed nest edge
is refused BY NAME, not by the generic "not ported" sentence.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_namelist_import import INPUT_TEXT, _pair


def test_config_accepts_3_and_its_menu_names_wsm3():
    from gpuwm import config

    assert 3 in config.MP_PHYSICS_ACCEPTED
    assert "3 (WSM3)" in config._MP_PHYSICS_SCHEMA_MENU


def test_the_importer_translates_3_to_3_without_substitution(tmp_path):
    """The ndown_1 refusal is gone: mp_physics = 3 imports natively."""
    from gpuwm.experiment import load_experiment
    from gpuwm.namelist_import import _MP_MAP, import_namelists

    assert _MP_MAP[3] == (3, "WSM3", "WSM3")
    inp = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 3, 3")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))
    assert not any(item.key == "mp_physics" for item in report.substitutions)
    output = tmp_path / "wsm3.toml"
    output.write_text(toml_text, encoding="utf-8")
    assert load_experiment(output).root.run.mp_physics == 3


def test_the_dispatcher_routes_3_to_the_wsm3_adapter(monkeypatch):
    import gpuwm.core.microphysics as micro
    import gpuwm.core.wsm3 as wsm3

    seen = {}

    def fake_apply(state, cfg, dt, *, refl_10cm_due=False):
        seen["call"] = (state, cfg.mp_physics, dt, refl_10cm_due)
        return "wsm3-result"

    monkeypatch.setattr(wsm3, "apply", fake_apply)

    class Cfg:
        mp_physics = 3

    state = object()
    assert micro._dispatch_scheme(state, Cfg(), 72.0,
                                  refl_10cm_due=True) == "wsm3-result"
    assert seen["call"] == (state, 3, 72.0, True)


def test_wrf_species_flags_for_wsm3_are_wrfs():
    """F_QI = F_QS = F_QG = false, warm_rain = false, radii declared.

    Breakage prevented: radiation that keys its cold-species conversion on
    these flags (module_ra_sw.F:193-203) reading WSM3's cold cloud as water.
    """
    from gpuwm.core import rrtmg_legacy, rrtmgp
    from gpuwm.core.kf import KFPhaseMode, kf_phase_mode_for_microphysics

    assert rrtmg_legacy.legacy_cloud_fraction_flags(3) == (False, False)
    assert not rrtmg_legacy.legacy_ice_active(3)
    assert rrtmg_legacy.legacy_scheme_has_req(3, 1) == (1, 1, 1)
    assert rrtmg_legacy.legacy_scheme_has_req(3, 0) == (0, 0, 0)
    scheme = rrtmgp.cloud_optics_scheme(3)
    assert scheme == "wsm3"
    assert not rrtmgp.scheme_is_ice_active(scheme)
    assert not rrtmgp.scheme_has_snow_species(scheme)
    assert scheme in rrtmgp.SCHEME_RADII_COUPLINGS
    # module_physics_init.F sets warm_rain only for Kessler; KF's cascade
    # (module_cu_kfeta.F:2599/:2607) then takes its no-F_QS arm for WSM3.
    assert kf_phase_mode_for_microphysics(3) \
        is kf_phase_mode_for_microphysics(0)
    assert kf_phase_mode_for_microphysics(3) is not KFPhaseMode.WARM_RAIN


def test_the_state_carries_wsm3s_registry_package_and_no_ice_classes():
    from gpuwm.config import RunConfig
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.core.physics_inventory import microphysics_scratch_slots

    cfg = RunConfig(nx=8, ny=6, nz=10, dx=1000.0, dy=1000.0, ztop=12000.0,
                    dt=6.0, run_seconds=60.0, moist=True, mp_physics=3)
    shapes = state_array_shapes(cfg)
    for name in ("qv", "qc", "qr", "effc", "effi", "effs"):
        assert name in shapes, name
    for name in ("qi", "qs", "qg"):
        assert name not in shapes, name
    assert dict(microphysics_scratch_slots(3)) == {
        "rainnc": "mp_rainnc", "rainncv": "mp_rainncv", "sr": "mp_sr",
        "snownc": "mp_snownc", "snowncv": "mp_snowncv"}


def test_a_mixed_nest_edge_with_wsm3_is_refused_by_name():
    """Same-scheme WSM3 nests resolve; a mixed edge names WSM3's reason."""
    from types import SimpleNamespace

    from gpuwm.core.microphysics_transition import (
        resolve_microphysics_transition)

    def cfg(mp):
        return SimpleNamespace(mp_physics=mp, moist=True, moist_cq=True,
                               nest_microphysics_transition="same-scheme-only")

    contract = resolve_microphysics_transition(cfg(3), cfg(3))
    assert contract.mixed is False
    for parent, child in ((3, 6), (6, 3)):
        with pytest.raises(ValueError, match="WSM3.*phase|phase.*WSM3"):
            resolve_microphysics_transition(cfg(parent), cfg(child))


def test_wrfinput_restores_wsm3_instead_of_refusing_it():
    from gpuwm.ingest import wrfinput

    assert wrfinput.SUPPORTED_MICROPHYSICS[3] == "WSM3"
    assert 3 not in wrfinput.UNSUPPORTED_MICROPHYSICS


def test_wsm3_reflectivity_diagnostic_splits_qr_at_0c():
    """WOOF's declared REFL_10CM for mp=3: qr is rain at or above 0 C and
    snow below, exactly the split module_mp_wsm3.F makes."""
    from gpuwm.core.refl import (REFL_10CM_INPUT_SPECIES,
                                 wsm3_precipitation_phases)

    assert REFL_10CM_INPUT_SPECIES[3] == ("qv", "qr")
    qr = np.asarray([1e-3, 2e-3, 3e-3, 4e-3], np.float32)
    t = np.asarray([280.0, 273.15, np.nextafter(np.float32(273.15),
                                                np.float32(0.0)), 250.0],
                   np.float32)
    rain, snow = wsm3_precipitation_phases(np, qr, t)
    zero = np.float32(0.0)
    np.testing.assert_array_equal(rain, [qr[0], qr[1], zero, zero])
    np.testing.assert_array_equal(snow, [zero, zero, qr[2], qr[3]])
    assert rain.dtype == np.float32 and snow.dtype == np.float32


def test_the_registry_publishes_wsm3_as_an_implemented_option():
    import json
    from pathlib import Path

    import gpuwm

    registry = json.loads((Path(gpuwm.__file__).parent
                           / "physics_registry_v2.json").read_text())
    option = registry["components"]["microphysics"]["options"]["wsm3-mp3"]
    assert option["implemented"] is True
    assert option["selectors"] == {"mp_physics": 3}
    rows = option["consumers"]
    assert rows["ring_guard"]["state_fields"] == [
        "thp", "qv", "qc", "qr", "effc", "effi", "effs"]
    assert rows["nest_transition"]["mixed_edge_ported"] is False
    assert rows["cloud_optics"]["rte_rrtmgp_coupling"] == "wsm3"
    assert rows["cloud_optics"]["legacy_declares_radii"] is True
    assert rows["vertical_level_bounds"]["maximum"] == 80


def test_wsm3_workspace_is_rebuilt_and_precipitation_is_serialized():
    """The real forecast's final digest must admit WSM3's per-call buffers.

    Unknown slots still refuse. Only the five accumulated/last-call surface
    values persist; recomputed geometry and thermodynamic work arrays do not.
    """
    from gpuwm.io.restart import classify_scratch_slot, RestartManifestError
    for suffix in ('theta','rho','pii','dz','z8w','w'):
        assert classify_scratch_slot('wsm3_'+suffix) == 'rebuild'
    for slot in ('mp_rainnc','mp_rainncv','mp_snownc','mp_snowncv','mp_sr'):
        assert classify_scratch_slot(slot) == 'serialize'
    with pytest.raises(RestartManifestError):
        classify_scratch_slot('unregistered_port_workspace')


def test_reflectivity_phase_workspace_is_transient_in_the_frame_digest():
    """The real CLI keeps REFL_10CM and discards recomputed phase buffers."""
    from types import SimpleNamespace
    from gpuwm.state_digest import _canonical_extra_manifest
    slots = {name: np.ones((2, 3, 4), np.float32) for name in
             ('refl_10cm', 'refl_wsm3_rain', 'refl_wsm3_snow', 'refl_wsm3_graupel')}
    state = SimpleNamespace(_scratch=slots)
    assert set(_canonical_extra_manifest(state)) == {'scratch/refl_10cm'}
    state._scratch['refl_wsm3_unregistered'] = np.zeros((2, 3, 4), np.float32)
    with pytest.raises(RuntimeError, match='not a concrete registered member'):
        _canonical_extra_manifest(state)


def test_wsm3_measured_frame_prices_both_launch_tiers():
    from gpuwm.core import preflight
    frame, select_tier = preflight.CONDITIONAL_COLUMN_TIERED_FRAMES['wsm3']
    assert frame.frame_bytes(select_tier(40)) == 5664
    assert frame.frame_bytes(select_tier(64)) == 5664
    assert frame.frame_bytes(select_tier(65)) == 7072
    assert frame.frame_bytes(select_tier(80)) == 7072
