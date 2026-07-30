"""Selection surface for the explicit legacy-RRTMG option (executable).

The exact port of WRF v4.6.1's bundled RRTMG is selectable as a peer of
the established (4,4) -> RTE+RRTMGP substitution: RunConfig field
``ra_rrtmg_variant``, compatibility token ``wrf-rrtmg-4-4-legacy-v1``,
distinct restart identity, and its own packaged coefficient assets.
With the integration wave landed, selecting it constructs the real
``gpuwm.core.rrtmg_legacy.RRTMGLegacyRadiation`` adapter (whose
constructor performs the readiness proof and fails closed if assets or
kernels are missing) -- never a silent fallback to RTE+RRTMGP.  The
byte-identity guarantees for the rte-rrtmgp default are unchanged.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.physics_compat import (
    RRTMG_VARIANT_LEGACY,
    RRTMG_VARIANT_RTE_RRTMGP,
    WRF_RRTMG_LEGACY,
    WRF_RRTMG_TO_RTE_RRTMGP,
    require_rrtmg_legacy_executable,
    require_rrtmg_legacy_ready,
    rrtmg_variant,
)


def _cfg(**updates):
    values = dict(nx=2, ny=1, nz=4, dx=1000.0, dy=1000.0,
                  ztop=8000.0, dt=10.0, run_seconds=60.0)
    values.update(updates)
    return RunConfig(**values)


def _require_gpu():
    """Adapter construction compiles CUDA kernels (GPU-light)."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA GPU available")


class _Stop(Exception):
    """Sentinel: halt initialize_physics right after adapter creation."""


def _recording_legacy_class(monkeypatch, constructed):
    """Swap in a subclass that records the REAL constructed adapter and
    then raises the sentinel (the SimpleNamespace state cannot support
    the rest of initialize_physics; the construction itself is real)."""
    import gpuwm.core.rrtmg_legacy as legacy

    real = legacy.RRTMGLegacyRadiation

    class Recorder(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            constructed.append(self)
            raise _Stop

    monkeypatch.setattr(legacy, "RRTMGLegacyRadiation", Recorder)
    return real


# ---------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------

def test_default_configs_stay_on_rte_rrtmgp_unchanged():
    cfg = validate_run_config(_cfg(ra_physics=4))
    assert cfg.ra_rrtmg_variant == RRTMG_VARIANT_RTE_RRTMGP
    assert rrtmg_variant(cfg) == RRTMG_VARIANT_RTE_RRTMGP
    # Pre-field configs (e.g. restored objects) resolve to the default.
    assert rrtmg_variant(SimpleNamespace()) == RRTMG_VARIANT_RTE_RRTMGP


def test_legacy_variant_is_accepted_on_the_44_pair():
    cfg = validate_run_config(_cfg(
        ra_physics=4, ra_rrtmg_variant=RRTMG_VARIANT_LEGACY))
    assert rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY
    split = validate_run_config(_cfg(
        ra_lw_physics=4, ra_sw_physics=4,
        ra_rrtmg_variant=RRTMG_VARIANT_LEGACY))
    assert rrtmg_variant(split) == RRTMG_VARIANT_LEGACY


def test_legacy_variant_requires_the_44_pair():
    with pytest.raises(ValueError, match="requires the.*4/4"):
        validate_run_config(_cfg(ra_rrtmg_variant=RRTMG_VARIANT_LEGACY))
    with pytest.raises(ValueError, match="requires the.*4/4"):
        validate_run_config(_cfg(
            ra_lw_physics=0, ra_sw_physics=1,
            ra_rrtmg_variant=RRTMG_VARIANT_LEGACY))


def test_variant_value_is_validated():
    with pytest.raises(ValueError, match="ra_rrtmg_variant must be"):
        validate_run_config(_cfg(
            ra_physics=4, ra_rrtmg_variant="approximate"))


def test_legacy_token_binds_to_the_legacy_variant():
    cfg = validate_run_config(_cfg(
        ra_physics=4, ra_rrtmg_variant=RRTMG_VARIANT_LEGACY,
        wrf_rrtmg_compatibility=WRF_RRTMG_LEGACY))
    assert cfg.wrf_rrtmg_compatibility == WRF_RRTMG_LEGACY
    # legacy token without the legacy variant is contradictory
    with pytest.raises(ValueError, match="requires\n?.*rrtmg_legacy"):
        validate_run_config(_cfg(
            ra_physics=4, wrf_rrtmg_compatibility=WRF_RRTMG_LEGACY))
    # RTE+RRTMGP substitution token cannot ride a legacy-variant config
    with pytest.raises(ValueError, match="contradicts"):
        validate_run_config(_cfg(
            ra_physics=4, ra_rrtmg_variant=RRTMG_VARIANT_LEGACY,
            wrf_rrtmg_compatibility=WRF_RRTMG_TO_RTE_RRTMGP))
    # legacy token still needs the 4/4 pair (the generalized token/pair
    # check fires first under the token-union rules)
    with pytest.raises(ValueError, match="4/4 pair"):
        validate_run_config(_cfg(
            wrf_rrtmg_compatibility=WRF_RRTMG_LEGACY))


def test_rrtmgp_substitution_token_still_validates_unchanged():
    cfg = validate_run_config(_cfg(
        ra_physics=4,
        wrf_rrtmg_compatibility=WRF_RRTMG_TO_RTE_RRTMGP))
    assert cfg.ra_rrtmg_variant == RRTMG_VARIANT_RTE_RRTMGP


# ---------------------------------------------------------------------
# Executable reality at the construction sites
# ---------------------------------------------------------------------

def test_selecting_legacy_constructs_the_adapter_at_physics_setup(
        monkeypatch):
    """No silent fallback: the (4,4)+legacy pair builds the REAL legacy
    adapter (its constructor's readiness proof runs) and never RRTMGP."""
    _require_gpu()
    import gpuwm.core.physics as physics
    import gpuwm.core.rrtmgp as rrtmgp

    cfg = _cfg(ra_physics=4, ra_rrtmg_variant=RRTMG_VARIANT_LEGACY)
    monkeypatch.setattr(physics, "physics_driver_required",
                        lambda _cfg: True)

    def _no_rrtmgp(*args, **kwargs):
        raise AssertionError(
            "RTE+RRTMGP adapter constructed under the legacy variant")

    monkeypatch.setattr(rrtmgp, "RRTMGPRadiation", _no_rrtmgp)
    constructed = []
    real = _recording_legacy_class(monkeypatch, constructed)
    with pytest.raises(_Stop):
        physics.initialize_physics(
            SimpleNamespace(), cfg,
            radiation_start_time=datetime(2001, 6, 15, 12),
            radiation_latitude=np.zeros((1, 2), np.float32),
            radiation_longitude=np.zeros((1, 2), np.float32))
    assert len(constructed) == 1
    adapter = constructed[0]
    assert isinstance(adapter, real)
    assert adapter.start_time == datetime(2001, 6, 15, 12)
    assert adapter.latitude_deg.shape == (1, 2)
    # construction is the readiness proof: identity is declared, and the
    # coefficient/table builds exist on the instance
    identity = adapter.restart_identity()
    assert identity["algorithms"]["lw"] == "wrf-v4.6.1-rrtmg-legacy-lw-v1"
    assert adapter._C is not None and adapter._sw_tables is not None


def test_readiness_helper_passes_here_and_receipts_when_broken(
        monkeypatch):
    """require_rrtmg_legacy_ready: silent on a complete installation,
    one complete receipt when anything is missing."""
    import gpuwm.physics_compat as compat

    require_rrtmg_legacy_ready()          # complete install: no raise
    assert require_rrtmg_legacy_executable is require_rrtmg_legacy_ready
    monkeypatch.setattr(
        compat, "_RRTMG_LEGACY_ASSETS",
        compat._RRTMG_LEGACY_ASSETS + ("data/wrf_radiation/NOT_A_FILE",),
        raising=True)
    with pytest.raises(NotImplementedError) as excinfo:
        compat.require_rrtmg_legacy_ready()
    message = str(excinfo.value)
    assert "NOT_A_FILE" in message
    assert "no silent fallback" in message


def test_every_rrtmgp_construction_site_has_a_legacy_peer():
    """Every module that constructs RRTMGPRadiation for a (4,4) request
    must construct RRTMGLegacyRadiation under the legacy variant."""
    import inspect

    import gpuwm.core.physics as physics
    import gpuwm.runtime as runtime

    for module in (physics, runtime):
        source = inspect.getsource(module)
        rrtmgp_sites = source.count("RRTMGPRadiation(")
        legacy_sites = source.count("RRTMGLegacyRadiation(")
        assert rrtmgp_sites > 0 and legacy_sites == rrtmgp_sites, (
            f"{module.__name__}: {rrtmgp_sites} RRTMGP construction(s) "
            f"vs {legacy_sites} legacy construction(s); every 4/4 site "
            "must serve both variants")


# ---------------------------------------------------------------------
# Restart identity and assets
# ---------------------------------------------------------------------

def test_restart_identity_strings_are_distinct_from_rte_rrtmgp():
    from gpuwm.io import restart

    assert restart.RRTMG_LEGACY_LW_ALGORITHM_IDENTITY == \
        "wrf-v4.6.1-rrtmg-legacy-lw-v1"
    assert restart.RRTMG_LEGACY_SW_ALGORITHM_IDENTITY == \
        "wrf-v4.6.1-rrtmg-legacy-sw-v1"
    assert restart.RRTMG_LEGACY_LW_ALGORITHM_IDENTITY != \
        restart.LONGWAVE_ALGORITHM_IDENTITIES[4]
    assert restart.RRTMG_LEGACY_SW_ALGORITHM_IDENTITY != \
        restart.SHORTWAVE_ALGORITHM_IDENTITIES[4]
    assert restart.RRTMG_LEGACY_ABOVE_ATMOSPHERE_POLICY != \
        restart.LONGWAVE_ABOVE_ATMOSPHERE_POLICIES[4]


def test_legacy_identity_resolves_distinct_slot_algorithms():
    from gpuwm.io import restart

    cfg = validate_run_config(_cfg(
        ra_physics=4, ra_rrtmg_variant=RRTMG_VARIANT_LEGACY))

    class DeclaredLegacyStub:
        start_time = datetime(2001, 6, 15, 12)
        latitude_deg = np.zeros((1, 2), np.float32)
        longitude_deg = np.zeros((1, 2), np.float32)
        restart_identity = {
            "algorithm": "stub-legacy-rrtmg",
            "above_atmosphere_policy": "stub-policy",
        }

    driver = SimpleNamespace(radiation_callable=DeclaredLegacyStub())
    identity = restart._radiation_setup_identity(driver, cfg)
    assert identity["algorithms"] == {
        "lw": restart.RRTMG_LEGACY_LW_ALGORITHM_IDENTITY,
        "sw": restart.RRTMG_LEGACY_SW_ALGORITHM_IDENTITY,
    }
    assert identity["above_atmosphere_policies"]["lw"] == \
        restart.RRTMG_LEGACY_ABOVE_ATMOSPHERE_POLICY
    # a would-be legacy callable is custom, never "stock RRTMGP"
    assert identity["callable"]["implementation"] == "custom"
    # the config fingerprint carries the variant, so a restart written
    # under one 4/4 implementation refuses to resume under the other
    fp_legacy = restart._configuration_fingerprint(cfg)
    fp_modern = restart._configuration_fingerprint(
        validate_run_config(_cfg(ra_physics=4)))
    assert fp_legacy != fp_modern


def test_legacy_assets_bind_the_packaged_rrtmg_data_files():
    from gpuwm.io import restart
    from gpuwm.ingest.rrtmg_coeffs import (RRTMG_LW_DATA_SHA256,
                                           RRTMG_SW_DATA_SHA256)

    cfg = validate_run_config(_cfg(
        ra_physics=4, ra_rrtmg_variant=RRTMG_VARIANT_LEGACY))
    identity = restart._active_asset_identity(cfg, None)
    assert identity["wrf_rrtmg_lw_data"]["sha256"] == RRTMG_LW_DATA_SHA256
    assert identity["wrf_rrtmg_sw_data"]["sha256"] == RRTMG_SW_DATA_SHA256
    # the modern variant does not carry the legacy coefficient assets
    modern = restart._active_asset_identity(
        validate_run_config(_cfg(ra_physics=4)), None)
    assert "wrf_rrtmg_lw_data" not in modern


# ---------------------------------------------------------------------
# Namelist import (peer mapping, byte-identical default)
# ---------------------------------------------------------------------

def test_import_maps_44_to_legacy_only_on_request(tmp_path):
    import test_namelist_import as tni
    from gpuwm.namelist_import import import_namelists

    wps, inp = tni._pair(tmp_path)
    default_text, default_report = import_namelists(wps, inp, name="synth")
    explicit_text, _ = import_namelists(
        wps, inp, name="synth", rrtmg_variant=RRTMG_VARIANT_RTE_RRTMGP)
    assert explicit_text == default_text, (
        "explicit rte-rrtmgp must be byte-identical to the default")
    # Post-assembly the modern importer default is the -v2 receipt (the
    # seam branch's snow-discount coupling); -v1 remains accepted for
    # committed historical configs but is never emitted fresh.
    assert 'wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"' \
        in default_text
    assert "ra_rrtmg_variant" not in default_text

    legacy_text, legacy_report = import_namelists(
        wps, inp, name="synth", rrtmg_variant=RRTMG_VARIANT_LEGACY)
    assert f'wrf_rrtmg_compatibility = "{WRF_RRTMG_LEGACY}"' in legacy_text
    assert 'ra_rrtmg_variant = "rrtmg_legacy"' in legacy_text
    # No substitution-family token of any version may appear in a
    # legacy-variant import.
    assert "wrf-rrtmg-4-4-to-rte-rrtmgp" not in legacy_text
    assert "WRF legacy RRTMG" in legacy_report.format()

    with pytest.raises(ValueError, match="rrtmg_variant must be"):
        import_namelists(wps, inp, name="synth", rrtmg_variant="modern")


def test_rrtmg_option_keys_are_ratified_with_failclosed_ranges(tmp_path):
    """cldovrlp/idcor/o3input/ghg_input/aer_opt: the campaign-pinned values
    import cleanly (dropped with a reason) for BOTH 4/4 variants and leave
    the emitted TOML byte-identical to an import without the keys; any
    other explicit value refuses instead of silently changing semantics."""
    import test_namelist_import as tni
    from gpuwm.namelist_import import import_namelists

    pinned = ("cldovrlp = 2,\n idcor = 0,\n o3input = 2,\n"
              " ghg_input = 0,\n aer_opt = 0,\n")
    inp_pinned = tni.INPUT_TEXT.replace("&physics\n", "&physics\n " + pinned)
    base_wps, base_inp = tni._pair(tmp_path)
    base_text, _ = import_namelists(base_wps, base_inp, name="synth")
    pinned_dir = tmp_path / "pinned"
    pinned_dir.mkdir()
    wps2, inp2 = tni._pair(pinned_dir, inp=inp_pinned)
    for variant in (RRTMG_VARIANT_RTE_RRTMGP, RRTMG_VARIANT_LEGACY):
        text, report = import_namelists(
            wps2, inp2, name="synth", rrtmg_variant=variant)
        formatted = report.format()
        for key in ("cldovrlp", "idcor", "o3input", "ghg_input", "aer_opt"):
            assert key in formatted, f"{key} drop must be receipted"
    rte_text, _ = import_namelists(
        wps2, inp2, name="synth", rrtmg_variant=RRTMG_VARIANT_RTE_RRTMGP)
    assert rte_text == base_text, (
        "pinned-value radiation keys must not perturb the emitted TOML")
    for key, bad in (("cldovrlp", 3), ("idcor", 1), ("o3input", 0),
                     ("ghg_input", 1), ("aer_opt", 1)):
        inp_bad = tni.INPUT_TEXT.replace(
            "&physics\n", f"&physics\n {key} = {bad},\n")
        inp3 = tmp_path / f"bad_{key}.input"
        inp3.write_text(inp_bad)
        with pytest.raises(ValueError, match=key):
            import_namelists(wps2, inp3, name="synth")


def test_imported_legacy_toml_round_trips_and_constructs(tmp_path,
                                                         monkeypatch):
    _require_gpu()
    import test_namelist_import as tni
    from gpuwm.experiment import load_experiment
    from gpuwm.namelist_import import import_namelists
    import gpuwm.core.physics as physics

    wps, inp = tni._pair(tmp_path)
    legacy_text, _ = import_namelists(
        wps, inp, name="synth", rrtmg_variant=RRTMG_VARIANT_LEGACY)
    out = tmp_path / "legacy.toml"
    out.write_text(legacy_text)
    exp = load_experiment(out)
    cfg = exp.root.run
    assert cfg.ra_rrtmg_variant == RRTMG_VARIANT_LEGACY
    assert cfg.wrf_rrtmg_compatibility == WRF_RRTMG_LEGACY
    monkeypatch.setattr(physics, "physics_driver_required",
                        lambda _cfg: True)
    constructed = []
    real = _recording_legacy_class(monkeypatch, constructed)
    with pytest.raises(_Stop):
        physics.initialize_physics(
            SimpleNamespace(), cfg,
            radiation_start_time=datetime(1999, 5, 3, 12),
            radiation_latitude=np.zeros((1, 2), np.float32),
            radiation_longitude=np.zeros((1, 2), np.float32))
    assert len(constructed) == 1 and isinstance(constructed[0], real)


def test_tree_radiation_workspace_selection_is_variant_aware():
    """The multi-domain build allocates the shared RRTMGP chunk workspace
    only for the modern variant.  Under ra_rrtmg_variant='rrtmg_legacy'
    the estimator's workspace_bytes is the legacy transient call-peak
    envelope (priced, never held; engine-default column_chunk=None), so
    constructing the modern workspace both wastes the allocation and
    trips the memory-ledger drift guard -- observed live on the first
    four-domain legacy launch (2026-07-28): "runtime RRTMGP workspace
    drifted from the memory ledger: 1025700000 != 2656051200 bytes".
    """
    from gpuwm.core.model import uses_modern_rrtmgp_workspace

    modern = _cfg(ra_physics=4)
    legacy = _cfg(ra_physics=4, ra_rrtmg_variant=RRTMG_VARIANT_LEGACY,
                  wrf_rrtmg_compatibility=WRF_RRTMG_LEGACY)
    dark = _cfg()

    def exp_of(*runs):
        return SimpleNamespace(domains=tuple(
            SimpleNamespace(run=run) for run in runs))

    assert uses_modern_rrtmgp_workspace(exp_of(modern, modern))
    assert not uses_modern_rrtmgp_workspace(exp_of(legacy, legacy))
    assert not uses_modern_rrtmgp_workspace(exp_of(dark, dark))
    # Radiation-dark domains beside 4/4 domains never flip the decision
    # either way: modern still builds, legacy still must not.
    assert uses_modern_rrtmgp_workspace(exp_of(modern, dark))
    assert not uses_modern_rrtmgp_workspace(exp_of(legacy, dark))
