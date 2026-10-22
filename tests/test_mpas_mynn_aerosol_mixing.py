"""MYNN number-species mixing split by family, CPU contract (no device).

RRFS v2 runs MYNN-EDMF with mixscalars=0, mixaerosols=1, mixnumcon=0
(rrfs-workflow 45ace1bf parm/hrrrv5/namelist.atmosphere:61-63), and the NOAA
MPAS fork adds MYNN's nwfa/nifa rates to the dynamics for aerosol-aware
Thompson and TEMPO (RRFSx/MPAS-Model 1d6e370e mpas_atmphys_todynamics.F
:384-405).  These tests hold the configuration table, the column seam's
generation rows, the hex carrier and the namelist importer to that.  The
device twin is tests/test_mpas_mynn_aerosol_mixing_gpu.py.
"""

from pathlib import Path

import numpy as np
import pytest

from gpuwm.config import (MYNN_QN_FAMILY_SCHEMES, MYNN_QN_FLAG_SPECIES,
                          MYNN_QN_MIX_KEYS, RunConfig,
                          mynn_mixed_number_species, mynn_mixscalars_active,
                          mynn_mixscalars_driver_value,
                          mynn_mixscalars_unported_species,
                          validate_run_config)

ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides):
    values = dict(nx=8, ny=8, nz=50, dx=3000.0, dy=3000.0,
                  dt=15.0, run_seconds=300.0, ztop=20000.0,
                  moist=True, mp_physics=28, bl_pbl_physics=5,
                  sf_sfclay_physics=5, sf_surface_physics=3,
                  num_soil_layers=9, bldt=0.0)
    values.update(overrides)
    return RunConfig(**values)


def test_the_keys_default_off_and_change_nothing():
    cfg = _config()
    assert (cfg.bl_mynn_mixaerosols, cfg.bl_mynn_mixnumcon) == (0, 0)
    assert mynn_mixed_number_species(cfg) == ()
    assert not mynn_mixscalars_active(cfg)
    assert mynn_mixscalars_driver_value(cfg) == 0


@pytest.mark.parametrize('keys, species, mp', [
    pytest.param({'bl_mynn_mixaerosols': 1}, ('nwfa', 'nifa'), 900, id='keys0-species0-900'),
])
def test_each_key_couples_its_own_family(mp, keys, species):
    cfg = _config(mp_physics=mp, **keys)
    validate_run_config(cfg)
    assert mynn_mixed_number_species(cfg) == species
    assert mynn_mixscalars_active(cfg)
    # The driver runs the whole family solve; the coupling keeps the subset.
    assert mynn_mixscalars_driver_value(cfg) == 1


def test_rrfs_v2_settings_mix_the_aerosols_only():
    cfg = _config(mp_physics=900, bl_mynn_mixscalars=0,
                  bl_mynn_mixaerosols=1, bl_mynn_mixnumcon=0)
    validate_run_config(cfg)
    assert mynn_mixed_number_species(cfg) == ("nwfa", "nifa")


def test_tempo_is_a_whole_family_scheme():
    # The latent defect this lane found: mp=900 had no row, so the table
    # read TEMPO as a scheme without number species and bl_mynn_mixscalars=1
    # was admitted and silently inert.
    assert 900 in MYNN_QN_FAMILY_SCHEMES
    assert MYNN_QN_FLAG_SPECIES[900] == ("qnc", "qni", "qnwfa", "qnifa")


def test_a_scheme_without_the_species_is_wrfs_no_op():
    for keys in ({"bl_mynn_mixaerosols": 1}, {"bl_mynn_mixnumcon": 1}):
        cfg = _config(mp_physics=6, **keys)
        validate_run_config(cfg)
        assert mynn_mixed_number_species(cfg) == ()
        assert mynn_mixscalars_driver_value(cfg) == 0
    # Thompson mp=8 carries no aerosol, so mixaerosols is a no-op there...
    cfg = _config(mp_physics=8, bl_mynn_mixaerosols=1)
    validate_run_config(cfg)
    assert mynn_mixscalars_unported_species(cfg) == ()


def test_a_partial_scheme_is_refused_by_species():
    # ...but its qni is mixed by mixnumcon in MYNN-EDMF, and this solve
    # cannot run on a state without the rest of the family.
    cfg = _config(mp_physics=8, bl_mynn_mixnumcon=1)
    assert mynn_mixscalars_unported_species(cfg) == ("qni",)
    with pytest.raises(NotImplementedError, match="bl_mynn_mixnumcon=1.*qni"):
        validate_run_config(cfg)


@pytest.mark.parametrize('value, key', [
    pytest.param(-1, 'bl_mynn_mixaerosols', id='-1-bl_mynn_mixaerosols'),
])
def test_each_key_is_an_integer_switch(key, value):
    with pytest.raises(ValueError, match=key):
        validate_run_config(_config(**{key: value}))


@pytest.mark.parametrize('key', ['bl_mynn_mixaerosols'])
def test_split_keys_need_mynn_and_exclude_scalar_pblmix(key):
    with pytest.raises(NotImplementedError, match=f"{key}=1 requires bl_pbl_physics=5"):
        validate_run_config(_config(bl_pbl_physics=1, sf_sfclay_physics=1,
                                    **{key: 1}))
    with pytest.raises(ValueError, match=f"scalar_pblmix=1 and {key}=1"):
        validate_run_config(_config(scalar_pblmix=1, **{key: 1}))


def test_the_restart_invariant_is_every_step_not_bldt_zero():
    # A PBL call on every step (WRF STEPBL=1) rewrites the held rates
    # before any read, exactly as bldt=0 does: admitted.
    validate_run_config(_config(bl_mynn_mixaerosols=1, bldt=15.0 / 60.0))
    # A longer cadence leaves mid-interval restarts without them: refused.
    with pytest.raises(NotImplementedError, match="every model step"):
        validate_run_config(_config(bl_mynn_mixaerosols=1, bldt=1.0))
    # The held mixscalars guard keeps its words.
    with pytest.raises(NotImplementedError, match="restart"):
        validate_run_config(_config(bl_mynn_mixscalars=1, bldt=60.0))


@pytest.mark.parametrize("key", ["bl_mynn_mixaerosols", "bl_mynn_mixnumcon"])
def test_default_off_keeps_old_restart_identity_and_on_is_bound(key):
    from dataclasses import asdict

    from gpuwm.io.restart import (RestartMismatchError,
                                  _drop_default_off_run_keys,
                                  _require_config_match)
    old = asdict(_config())
    old.pop(key)
    off = asdict(_config())
    on = asdict(_config(**{key: 1}))
    for values in (old, off, on):
        _drop_default_off_run_keys(values)
    assert old == off
    assert on.pop(key) == 1
    assert on == off
    _require_config_match(old, _config(), "off-checkpoint")
    with pytest.raises(RestartMismatchError, match=key):
        _require_config_match(old, _config(**{key: 1}), "off-checkpoint")


def test_the_column_suite_carries_the_split_keys_on_mynn_only():
    from gpuwm.core import mpas_column_suite as suite
    for key in ("bl_mynn_mixaerosols", "bl_mynn_mixnumcon"):
        row = suite.GENERATION_SELECTORS[key]
        assert row["values"] == (0, 1)
        assert row["requires"] == ("pbl_scheme", ("mynn",))
    selection = dict(radiation_scheme="rrtmg_legacy",
                     surface_layer_scheme="mynn", land_surface_scheme="ruc",
                     pbl_scheme="mynn", drag_scheme="gsl")
    assert suite.resolve_generation(
        {"bl_mynn_mixaerosols": 1}, microphysics_scheme="tempo",
        selection=selection) == {"bl_mynn_mixaerosols": 1}
    with pytest.raises(ValueError, match="recorded and never read"):
        suite.resolve_generation(
            {"bl_mynn_mixaerosols": 1}, microphysics_scheme="tempo",
            selection=dict(selection, pbl_scheme="ysu",
                           surface_layer_scheme="revised_mm5"))


def test_the_suite_result_carries_extra_rates_and_the_frozen_contract_is_unchanged():
    from gpuwm.core.mpas_column_batch import MpasColumnPhysicsTendencies
    from gpuwm.core.mpas_column_suite import MpasColumnSuiteTendencies
    names = ("du", "dv", "dtheta", "dqv", "dqc", "dqr", "dqi", "dqs", "dqg",
             "h_diabatic")
    values = dict(**{name: None for name in names}, step_index=0,
                  elapsed_seconds=0.0, radiation_ran=False,
                  surface_pbl_ran=False, cumulus_ran=False)
    assert "scalar_rates" not in MpasColumnPhysicsTendencies.__dataclass_fields__
    result = MpasColumnSuiteTendencies(**values)
    assert isinstance(result, MpasColumnPhysicsTendencies)
    assert dict(result.scalar_rates) == {}


def test_the_hex_carrier_admits_row_species_rates_and_refuses_others():
    pytest.importorskip("hexcore")
    from hexcore.cuda_physics_v841 import (
        CudaPhaseOneExecutionProvenanceV841, CudaRawColumnPhysicsV841)
    from hexcore.species_row import registered_species_rows
    tempo = registered_species_rows().get("mp_tempo")
    if tempo is None:
        pytest.skip("this tree registers no TEMPO species row")
    names = tempo.names()
    zero = np.zeros((2, 3), np.float32)
    masses = {name: zero for name in ("qv", "qc", "qr", "qi", "qs", "qg")}
    provenance = CudaPhaseOneExecutionProvenanceV841.arwen_gwd_off(
        aggregate_executed=True)
    raw = CudaRawColumnPhysicsV841(
        du=zero, dv=zero, dtheta=zero,
        dscalars={**masses, "nwfa": zero, "nifa": zero},
        time_seconds=0.0, execution_provenance=provenance,
        scalar_names=names)
    assert set(raw.dscalars) == set(masses) | {"nwfa", "nifa"}
    with pytest.raises(ValueError, match="outside"):
        CudaRawColumnPhysicsV841(
            du=zero, dv=zero, dtheta=zero,
            dscalars={**masses, "nbca": zero},
            time_seconds=0.0, execution_provenance=provenance,
            scalar_names=names)


def test_rrfs_v2s_namelist_aerosol_mixing_reaches_the_generation():
    pytest.importorskip("hexcore")
    pytest.importorskip("gpuwm.core.mpas_column_suite")
    from hexcore.physics_namelist import import_physics_namelist
    text = (ROOT / "tests/data/"
            "rrfs-v2-hrrrv5-namelist.atmosphere").read_text(encoding="utf-8")
    # The workflow's own 12 km placeholders, rendered as
    # exrrfs_fcst.sh renders them.
    for key, value in {
            "dt": "60.0", "substeps": "2",
            "start_time": "2026-10-03_12:00:00", "run_duration": "01:00:00",
            "pio_num_iotasks": "1", "pio_stride": "1",
            "MESH_NAME": "conus12km", "do_DAcycling": "false", "radt": "30",
            "lsm_scheme": "sf_ruc", "nsoillevels": "9", "jedi_da": "false",
            "do_sppt": "false"}.items():
        text = text.replace("${%s}" % key, value)
    from hexcore.config_v841 import LegacyEpssmIgnored
    # RRFS sets config_epssm = 0.1, which MPAS-A v8.4.1 reads and ignores with
    # LegacyEpssmIgnored.  It is asserted here so it cannot escape the test:
    # its class lives in the hex tree, which a pytest-xdist controller neither
    # imports nor can rebuild from the message, and the escaped warning aborted
    # the whole parallel CPU run (INTERNALERROR KeyError on the worker).
    with pytest.warns(LegacyEpssmIgnored, match="config_epssm=0.1 is ignored"):
        result = import_physics_namelist(text, mesh="conus3km")
        physics = result.require_admitted()
    statuses = {(d.option, d.status) for d in result.dispositions}
    assert ("config_mynn_mixaerosols", "admitted") in statuses
    assert ("config_mynn_mixnumcon", "admitted") in statuses
    assert "config_mynn_mixaerosols" not in {r.knob for r in result.refusals}
    assert physics.generation["bl_mynn_mixaerosols"] == 1
    # mixnumcon = 0 is the engine default and selects nothing.
    assert "bl_mynn_mixnumcon" not in physics.generation
