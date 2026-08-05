# tests/test_restart.py
"""Restart files: manifest completeness, exact roundtrip, bit-identity.

CPU tier (no device work): the DomainState/PhysicsDriver/scratch manifest
classification is exhaustively cross-checked against the source (a new
model attribute or scratch slot without a manifest entry fails here and in
every ``write_restart``), synthetic host arrays round-trip bit-exactly
through the NPZ format, and config/setup mismatches are rejected loudly.
The DomainState introspection runs on a NumPy-backed cupy shim so no GPU
allocation happens.

GPU tier (controller-run): a short full-physics 20+20-vs-40-step
continuation on the small physics state pins FP32 bit identity for every
serialized field, and the slow-acceptance real74 gate runs 6 h -> restart
-> FRESH PROCESS -> 6 h against the uninterrupted 12 h via the CLI,
comparing the complete 12 h restart serializations (every prognostic,
physics field, accumulator, and held tendency), the hour-7..12 wrfouts,
and the run-summary trackers.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import struct
import subprocess
import sys
from types import SimpleNamespace
import zipfile
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from gpuwm.config import RunConfig
from gpuwm.core.nssl2_contract import (
    CONTRACT_ID as NSSL2_CONTRACT_ID,
    DEFAULT_MODE as NSSL2_DEFAULT_MODE,
    WRF_NAMELIST_DEFAULTS as NSSL2_WRF_NAMELIST_DEFAULTS,
    WRF_REFERENCE_COMMIT as NSSL2_WRF_REFERENCE_COMMIT,
    WRF_REFERENCE_VERSION as NSSL2_WRF_REFERENCE_VERSION,
)
from gpuwm.io import restart


BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
requires_bundle = pytest.mark.skipif(
    not (BUNDLE / "wrfout_reference" /
         "wrfout_d01_1974-04-03_13_00_00").is_file(),
    reason="WRF_1974_MP55 reference bundle not present",
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# CPU helpers: a NumPy-backed DomainState (no device allocation).
# ---------------------------------------------------------------------------

class _NumpyCupyShim:
    """The subset of the cupy module surface DomainState allocation uses."""

    ndarray = np.ndarray
    float32 = np.float32

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(shape, dtype=np.float32):
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def asarray(value, dtype=None):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def ascontiguousarray(value):
        return np.ascontiguousarray(value)

    isfinite = staticmethod(np.isfinite)
    any = staticmethod(np.any)
    maximum = staticmethod(np.maximum)
    where = staticmethod(np.where)


class _DeclaredPhysicsCallable:
    """Minimal public injection-contract callable with explicit identity."""

    def __init__(self, name):
        self.restart_identity = {
            "algorithm": name,
            "above_atmosphere_policy": "declared-custom-policy-v1",
            "assets": {"fixture": {"sha256": "declared-by-caller"}},
        }

    def __call__(self, **kwargs):
        raise AssertionError("identity tests never execute physics")


def _cfg(**overrides) -> RunConfig:
    values = dict(nx=6, ny=4, nz=5, dx=2000.0, dy=2000.0, ztop=8000.0,
                  dt=10.0, run_seconds=0.0)
    values.update(overrides)
    return RunConfig(**values)


def _shim_state(cfg, monkeypatch):
    import gpuwm.core.state as state_module

    monkeypatch.setattr(state_module, "cp", _NumpyCupyShim)
    return state_module.DomainState(cfg)


def _fill_setup(state) -> None:
    """Deterministic nonzero setup arrays (identical across builds)."""
    for index, name in enumerate(restart.STATE_SETUP_ARRAYS):
        array = getattr(state, name)
        array[...] = np.arange(array.size, dtype=np.float64).reshape(
            array.shape).astype(np.float32) * np.float32(0.01 * (index + 1))


def _fill_serialized(state, seed: int) -> None:
    """Random FP32 payloads plus adversarial bit patterns."""
    rng = np.random.default_rng(seed)
    for name in restart.STATE_SERIALIZED_ATTRS:
        array = getattr(state, name, None)
        if array is None:
            continue
        array[...] = rng.standard_normal(array.shape).astype(np.float32)
        flat = array.reshape(-1)
        flat[0] = np.float32(-0.0)           # signed zero
        flat[-1] = np.float32(1.0e-42)       # denormal
        if flat.size > 2:
            flat[1] = np.float32(np.nan)     # NaN payload survives NPZ


def _replace_restart_member(path, output, key, replacement):
    """Rewrite one test member and keep the archive's self-manifest valid."""
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))
    payload[key] = np.asarray(replacement)
    header["array_manifest"][key] = {
        "shape": list(payload[key].shape), "dtype": str(payload[key].dtype)}
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    with output.open("wb") as stream:
        np.savez(stream, **payload)
    return output


def _rewrite_restart_archive(path, output, edit):
    """Apply a test-only archive edit and rebuild its self-manifest."""
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    header = json.loads(bytes(bytearray(
        payload.pop(restart._HEADER_KEY))).decode("utf-8"))
    edit(payload, header)
    header["array_manifest"] = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in payload.items()
    }
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    with output.open("wb") as stream:
        np.savez(stream, **payload)
    return output


def _shim_driver_state(cfg, monkeypatch):
    """NumPy-backed PhysicsDriver with no scheme callables or surface fields."""
    import gpuwm.core.physics as physics

    state = _shim_state(cfg, monkeypatch)
    monkeypatch.setattr(physics, "cp", _NumpyCupyShim)
    driver = physics.PhysicsDriver(
        state, cfg, fields={}, sfclay_result=None, noah_params=None)
    state.physics = driver
    return state, driver


def _identity_bound_physics_state(cfg, monkeypatch, *, trace_co2=3.30e-4,
                                  latitude_offset=0.0,
                                  noah_offset=0.0):
    """CPU-only fully configured physics setup for restart-identity tests."""
    from gpuwm.core.kf import KainFritsch
    from gpuwm.core.noah import NoahParams
    from gpuwm.core.rrtmgp import RRTMGPRadiation

    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    shape = state.mup.shape
    latitude = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    radiation = object.__new__(RRTMGPRadiation)
    radiation.start_time = datetime(1974, 4, 3, 12)
    radiation.latitude_deg = latitude + np.float32(latitude_offset)
    radiation.longitude_deg = np.full(shape, -87.0, dtype=np.float32)
    radiation.column_chunk = 17
    radiation.validation_mode = "fused"
    radiation.trace_gas_overrides = {"co2": 3.30e-4}
    radiation.trace_vmr = {"co2": float(trace_co2), "n2o": 3.0e-7}
    radiation._ozone_logp = np.linspace(1.0, 2.0, cfg.nz,
                                       dtype=np.float32)
    radiation._ozone_vmr = np.linspace(3.0e-8, 8.0e-6, cfg.nz,
                                      dtype=np.float32)
    radiation.update_count = 0
    radiation.lw_tables = SimpleNamespace(
        kind="lw", coefficient=np.array([1.0, 2.0], dtype=np.float64))
    radiation.sw_tables = SimpleNamespace(
        kind="sw", coefficient=np.array([3.0, 4.0], dtype=np.float64))
    radiation.lw_cloud_tables = SimpleNamespace(
        kind="lw", extinction=np.array([5.0], dtype=np.float64))
    radiation.sw_cloud_tables = SimpleNamespace(
        kind="sw", extinction=np.array([6.0], dtype=np.float64))
    radiation.chunk_workspace = None
    driver.radiation_callable = radiation
    driver.cumulus_callable = KainFritsch()
    driver.noah_params = NoahParams(
        veg=np.arange(30, dtype=np.float64).reshape(2, 15)
            + float(noah_offset),
        soil=np.arange(20, dtype=np.float64).reshape(2, 10),
        gen=np.arange(16, dtype=np.float64),
        lucats=2, slcats=2, bare=1, natural=2,
        lutype="USGS", sltype="STAS",
    )
    return state, driver


# ---------------------------------------------------------------------------
# Manifest completeness (CPU).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("overrides", [
    dict(moist=False),
    dict(moist=True, mp_physics=1),
    dict(moist=True, mp_physics=8),
    dict(moist=True, mp_physics=10),
    dict(moist=True, mp_physics=18),
], ids=["dry", "kessler", "thompson", "morrison", "nssl2"])
def test_every_domainstate_attribute_is_classified(monkeypatch, overrides):
    """A DomainState attribute outside the manifest raises; the serialized
    manifest names exactly the allocated cross-step arrays per config."""
    state = _shim_state(_cfg(**overrides), monkeypatch)
    for name in vars(state):
        restart.classify_state_attr(name)  # raises on any unclassified attr

    manifest = restart.state_manifest(state)
    expected = {f"state/{name}" for name in restart.STATE_SERIALIZED_ATTRS
                if getattr(state, name, None) is not None}
    assert set(manifest) == expected
    if overrides.get("mp_physics") == 10:
        # W6 advisory: h_diabatic is restart state (WRF `rdu`,
        # Registry.EM_COMMON:1389); effective radii feed the next
        # radiation call.
        assert {"state/h_diabatic", "state/effc", "state/ng"} <= set(manifest)
    if overrides.get("mp_physics") == 18:
        expected_nssl = {
            *(f"state/{name}" for name in restart.NSSL2_RESTART_PROGNOSTICS),
            *(f"state/{name}"
              for name in restart.NSSL2_RESTART_AUXILIARY_STATE),
        }
        assert expected_nssl <= set(manifest)
        assert not (set(manifest) & {
            "state/nc", "state/nr", "state/ni", "state/ns", "state/ng",
        })
    if not overrides.get("moist"):
        assert "state/qv" not in manifest


def test_unclassified_state_attribute_fails_the_write(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    state.brand_new_accumulator = np.zeros(3, dtype=np.float32)
    with pytest.raises(restart.RestartManifestError,
                       match="brand_new_accumulator"):
        restart.write_restart(tmp_path / "rst.npz", state, cfg)


def test_unclassified_scratch_slot_fails_the_write(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    state.scratch((2, 2), "mystery_buffer")
    with pytest.raises(restart.RestartManifestError, match="mystery_buffer"):
        restart.write_restart(tmp_path / "rst.npz", state, cfg)


def test_physicsdriver_attribute_classification_matches_source():
    """Every ``self.*`` assigned anywhere in PhysicsDriver is classified,
    and every classified name exists in the source (no stale entries)."""
    import gpuwm.core.physics as physics

    tree = ast.parse(Path(physics.__file__).read_text(encoding="utf-8"))
    class_node = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.ClassDef)
                      and node.name == "PhysicsDriver")
    assigned = set()
    for node in ast.walk(class_node):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            for sub in ast.walk(target):
                if (isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id == "self"):
                    assigned.add(sub.attr)
    classified = (set(restart.DRIVER_SERIALIZED_ATTRS)
                  | set(restart.DRIVER_REBUILT_ATTRS))
    assert "refl_10cm" in restart.DRIVER_REBUILT_ATTRS
    assert "refl_10cm" not in restart.DRIVER_SERIALIZED_ATTRS
    for name in ("_sr_roundoff_upper", "_sr_roundoff_max_ulps",
                 "_wsm6_minor_loops"):
        assert name in restart.DRIVER_REBUILT_ATTRS
        assert name not in restart.DRIVER_SERIALIZED_ATTRS
    assert assigned == classified, {
        "unclassified (add to gpuwm/io/restart.py)":
            sorted(assigned - classified),
        "stale manifest entries": sorted(classified - assigned),
    }


def _harvest_scratch_slots() -> tuple[set[str], set[str]]:
    """All literal slot names and dynamic-slot prefixes in the package."""
    import gpuwm

    literals: set[str] = set()
    prefixes: set[str] = set()
    for source in Path(gpuwm.__file__).parent.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "scratch"
                    and len(node.args) >= 2):
                continue
            slot = node.args[1]
            if isinstance(slot, ast.Constant) and isinstance(slot.value, str):
                literals.add(slot.value)
            elif isinstance(slot, ast.JoinedStr) and slot.values and \
                    isinstance(slot.values[0], ast.Constant):
                prefixes.add(str(slot.values[0].value))
            elif isinstance(slot, ast.BinOp) and \
                    isinstance(slot.left, ast.Constant):
                prefixes.add(str(slot.left.value))
            # Name-valued slots (dycore/diffusion spec tuples) are string
            # literals elsewhere in the same module; the module-wide
            # literal sweep below picks them up.
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and (node.value.startswith(("smag_", "diff6_", "diff_"))
                         )):
                literals.add(node.value)
    return literals, prefixes


def test_every_source_scratch_slot_is_classified():
    """New scratch slots must be classified: every literal slot name in
    the package classifies without raising, and every dynamically built
    slot family resolves to a known classification.

    Tier note (review F6): this static harvest covers direct literals,
    f-string/concatenation prefixes, and the Name-passed slot families via
    the module-wide constant sweep (smag_/diff6_/diff_).  A future
    Name-passed slot under some other prefix escapes THIS test only —
    the enforcement boundary is write_restart's runtime walk over
    ``state._scratch`` (every live slot classifies or the write raises),
    which the GPU gates execute against live full-physics state.  This
    test is the early-warning tier, not the enforcement."""
    literals, prefixes = _harvest_scratch_slots()
    assert "mp_rainnc" in literals and "cu_nca" in literals  # harvest sanity
    assert "smag_ru" in literals and "diff6_m" in literals   # name-passed
    for slot in sorted(literals):
        restart.classify_scratch_slot(slot)   # raises on any new slot
    for prefix in sorted(prefixes):
        assert (prefix.startswith(tuple(restart.REBUILT_SCRATCH_PREFIXES))
                or prefix == "cu_"), prefix
    # The one dynamic serialized family: the KF stored per-column rates.
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                 "rqrcuten", "rqscuten"):
        assert restart.classify_scratch_slot(f"cu_{name}") == "serialize"


def test_scratch_classification_kinds():
    for slot in ("mp_rainnc", "mp_snownc", "mp_graupelnc", "mp_sr",
                 "mp_kessler_sr", "cu_rainc", "cu_nca", "cu_pratec",
                 "cu_raincv", "cu_rthcuten"):
        assert restart.classify_scratch_slot(slot) == "serialize", slot
    for slot in ("rk_ww", "adv_ru", "smag_rqv", "smag_km", "diff6_m",
                 "diff_u", "acoustic_c2a", "openbc_upp_faces",
                 "moist_pd_q0", "pd_fxl", "morr_theta", "mp_th", "mp_z8w",
                 "nssl2_fused_temperature", "nssl2_primary_ice_target",
                 "nssl2_nucond_ss",
                 "refl_10cm", "physics_qtot", "lbc_relax_u",
                 "lbc_weights_0", "lbc_old_mup_frame_1",
                 "lbc_forcing_tables", "integration_health_partial"):
        assert restart.classify_scratch_slot(slot) == "rebuild", slot
    with pytest.raises(restart.RestartManifestError):
        restart.classify_scratch_slot("totally_new_slot")


def test_serialized_dataclasses_match_their_component_manifests():
    """Review F1: the component tuples are pinned to the dataclasses both
    ways, so a future PhysicsTendencies/MicrophysicsDiagnostics field
    without a manifest update fails here (and every write fails through
    the same helper)."""
    import dataclasses as dc

    from gpuwm.core.microphysics import MicrophysicsDiagnostics
    from gpuwm.core.physics import PhysicsTendencies

    assert ({field.name for field in dc.fields(PhysicsTendencies)}
            == set(restart.TENDENCY_COMPONENTS))
    assert ({field.name for field in dc.fields(MicrophysicsDiagnostics)}
            == set(restart.MICROPHYSICS_COMPONENTS))
    # The exact helper the write path runs accepts the real classes.
    restart._require_dataclass_components(
        PhysicsTendencies, restart.TENDENCY_COMPONENTS, "PhysicsTendencies")
    restart._require_dataclass_components(
        MicrophysicsDiagnostics, restart.MICROPHYSICS_COMPONENTS,
        "MicrophysicsDiagnostics")


def test_planted_dataclass_field_fails_the_write_helper():
    """A planted (unclassified) dataclass field must fail the write —
    the negative for review F1, in both directions."""
    import dataclasses as dc

    planted = dc.make_dataclass(
        "PlantedTendencies",
        [(name, object) for name in restart.TENDENCY_COMPONENTS]
        + [("rqg", object)])
    with pytest.raises(restart.RestartManifestError, match="rqg"):
        restart._require_dataclass_components(
            planted, restart.TENDENCY_COMPONENTS, "PhysicsTendencies")
    with pytest.raises(restart.RestartManifestError, match="gone_field"):
        restart._require_dataclass_components(
            planted,
            tuple(restart.TENDENCY_COMPONENTS) + ("rqg", "gone_field"),
            "PhysicsTendencies")


def test_callable_walk_catches_container_hidden_state():
    """Review F2: arrays hidden in dict-valued or object-container
    callable attributes fail the write unless the container is
    explicitly classified."""

    class Bag:
        pass

    scheme = Bag()
    scheme.w0avg = np.zeros((2, 2), np.float32)
    restart._callable_state_check(
        scheme, frozenset({"w0avg"}), frozenset(), "cumulus")

    scheme.rates = {"rthcuten": np.zeros(3, np.float32)}
    with pytest.raises(restart.RestartManifestError, match="rates"):
        restart._callable_state_check(
            scheme, frozenset({"w0avg"}), frozenset(), "cumulus")
    del scheme.rates

    tables = Bag()
    tables.k_major = np.zeros(3, np.float32)
    scheme.tables = tables
    with pytest.raises(restart.RestartManifestError, match="tables"):
        restart._callable_state_check(
            scheme, frozenset({"w0avg"}), frozenset(), "radiation")
    # Explicit container classification accepts it DELIBERATELY.
    restart._callable_state_check(
        scheme, frozenset({"w0avg"}), frozenset({"tables"}), "radiation")

    direct = Bag()
    direct.cache = np.zeros(3, np.float32)
    with pytest.raises(restart.RestartManifestError, match="cache"):
        restart._callable_state_check(
            direct, frozenset(), frozenset(), "radiation")

    scalars = Bag()
    scalars.trace_vmr = {"co2": 330.0e-6}     # scalar dicts pass
    scalars.start_time = datetime(1974, 4, 3, 12)
    scalars.update_count = 3
    restart._callable_state_check(
        scalars, frozenset(), frozenset(), "radiation")


def test_rrtmgp_table_containers_are_explicitly_classified():
    """The RRTMGP gas/cloud table objects (rebuild-on-load lru caches)
    are in the container allowlist by name, not silently invisible."""
    assert restart.RADIATION_CALLABLE_CONTAINERS == frozenset(
        {"lw_tables", "sw_tables", "lw_cloud_tables", "sw_cloud_tables",
         "chunk_workspace",
         # legacy-RRTMG adapter containers (all rebuild-on-load; see the
         # classification comments in gpuwm/io/restart.py)
         "_C", "_sw_tables", "_cuda_sw", "_ozone_climo",
         "_night_outputs", "_ozone"})
    assert restart.CUMULUS_CALLABLE_CONTAINERS == frozenset(
        {"_history_state", "_driver"})


def test_every_cumulus_adapter_attribute_is_classified():
    """Each selectable cumulus scheme's LIVE adapter passes the walk.

    The classification allowlists are written against the adapters, so the
    adapters -- not hand-built stand-ins -- are what has to satisfy them.
    The GF adapter is the reason this test exists: it stores the
    PhysicsDriver back-reference ``bind_driver`` hands it, the driver is an
    array-bearing object container, and the walk refused every checkpoint
    of a ``cu_physics=3`` run until ``_driver`` was classified (the first
    real-case GF trajectory died at its first restart having integrated
    179 outer steps).  Constructing each adapter here needs no device.
    """
    from gpuwm.core.gf import GrellFreitas
    from gpuwm.core.kf import KainFritsch

    def check(adapter):
        restart._callable_state_check(
            adapter, restart.CUMULUS_CALLABLE_ARRAYS,
            restart.CUMULUS_CALLABLE_CONTAINERS, "cumulus")

    check(KainFritsch())
    check(GrellFreitas())

    # Bound, which is the only shape a checkpoint ever sees: PhysicsDriver
    # calls bind_driver on every cumulus step, so a GF adapter that has run
    # once always carries the driver.
    driver_like = SimpleNamespace(
        rthratenlw=np.zeros((2, 2, 2), np.float32),
        rthratensw=np.zeros((2, 2, 2), np.float32))
    bound = GrellFreitas()
    bound.bind_driver(driver_like)
    check(bound)

    # The walk is still armed: an adapter that grows real state of its own
    # must be classified, not absorbed by the container entries above.
    grown = GrellFreitas()
    grown.closure_memory = np.zeros((4, 4), np.float32)
    with pytest.raises(restart.RestartManifestError,
                       match="closure_memory"):
        check(grown)


def test_surface_audit_carry_in_names_live_in_the_fields_inventory():
    """The audit's explicit restart list is covered by whole-dict fields
    serialization: every name is a member of the driver fields inventory
    (SFCLAY outputs + the Noah launch set)."""
    from gpuwm.core.noah import _F2D, _F3D
    from gpuwm.core.sfclay import SFCLAY_OUTPUTS

    inventory = set(_F2D) | set(_F3D) | set(SFCLAY_OUTPUTS) | {"pblh"}
    audit_list = {"ust", "mol", "znt", "qsfc", "hfx", "qfx", "pblh",
                  "sh2o", "snotime", "albedo", "emiss"}
    assert audit_list <= inventory


# ---------------------------------------------------------------------------
# Roundtrip and rejection (CPU).
# ---------------------------------------------------------------------------

def test_physics_storage_aliases_are_lifetime_gated(monkeypatch):
    import gpuwm.core.physics as physics
    from gpuwm.core.microphysics import MicrophysicsDiagnostics

    cfg = _cfg(moist=True, mp_physics=10, bl_pbl_physics=1,
               ra_physics=90, bldt=0.0)
    state, driver = _shim_driver_state(cfg, monkeypatch)

    for component, slot in physics.microphysics_scratch_slots(cfg.mp_physics):
        assert getattr(driver.microphysics, component) is state._scratch[slot]
    shape = state.mup.shape
    driver.accept_microphysics(MicrophysicsDiagnostics(
        rainnc=np.full(shape, 2.0, np.float32),
        rainncv=np.full(shape, 0.25, np.float32),
        sr=np.full(shape, 0.5, np.float32)))
    assert driver.microphysics.rainnc is state._scratch["mp_rainnc"]
    np.testing.assert_array_equal(
        driver.microphysics.rainnc,
        np.full_like(driver.microphysics.rainnc, 2.0))
    np.testing.assert_array_equal(
        driver._pending_rainbl, np.full_like(driver._pending_rainbl, 0.25))
    assert driver.tendencies is driver.pbl_tendencies
    assert not physics.physics_retains_ysu_output(cfg)
    assert physics.physics_reuses_pbl_composition(cfg)

    for array in vars(driver.pbl_tendencies).values():
        if array is not None:
            array[...] = np.float32(1.0)
    for array in vars(driver.radiation_tendencies).values():
        if array is not None:
            array[...] = np.float32(2.0)
    for array in vars(driver.cumulus_tendencies).values():
        if array is not None:
            array[...] = np.float32(3.0)
    driver._compose_tendencies(cfg)
    assert driver.tendencies is driver.pbl_tendencies
    np.testing.assert_array_equal(
        driver.tendencies.ru, np.full_like(driver.tendencies.ru, 1.0))
    for name in ("rtheta", "rqv", "rqc"):
        value = getattr(driver.tendencies, name)
        np.testing.assert_array_equal(value, np.full_like(value, 6.0))

    held_cfg = replace(cfg, bldt=5.0)
    _held_state, held_driver = _shim_driver_state(held_cfg, monkeypatch)
    assert held_driver.tendencies is not held_driver.pbl_tendencies
    assert physics.physics_retains_ysu_output(held_cfg)
    assert not physics.physics_reuses_pbl_composition(held_cfg)


def test_mixed_phase_kf_held_inventory_is_eager_before_first_due_call(
        monkeypatch):
    """KF phase categories are canonical held state, not lazy work arrays.

    The real74 driver reuses the fresh PBL stack as its composed target at
    ``bldt=0``.  Consequently mixed-phase KF requires QR/QI/QS storage both
    in the raw cumulus family and in that target before the first scheduled
    KF call.  A due call may change bytes, but it must not change the restart
    manifest's array inventory.
    """
    from gpuwm.core.physics import CumulusResult
    from gpuwm.verify.cases.real74_d02 import (
        canonical_state_digest, validate_inventory_growth)

    cfg = _cfg(moist=True, mp_physics=10, bl_pbl_physics=1,
               cu_physics=1, bldt=0.0)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    shape = state.p.shape
    clock = SimpleNamespace(dtbc_fp32=np.float32(cfg.dt))

    before = set(restart._driver_manifest(driver))
    before_inventory = canonical_state_digest(
        state, clock)["inventory"]
    for owner in ("pbl_tendencies", "cumulus_tendencies"):
        for component in ("rqr", "rqi", "rqs"):
            assert f"driver/{owner}/{component}" in before

    rates = {
        name: np.full(shape, index + 1.0, dtype=np.float32)
        for index, name in enumerate((
            "rthcuten", "rqvcuten", "rqccuten", "rqicuten",
            "rqrcuten", "rqscuten"))
    }
    driver.cumulus_callable = lambda **_kwargs: CumulusResult(**rates)
    driver._run_cumulus({}, state, cfg)
    driver._compose_tendencies(cfg)

    after = set(restart._driver_manifest(driver))
    assert after == before
    after_inventory = canonical_state_digest(state, clock)["inventory"]
    assert after_inventory["sha256"] == before_inventory["sha256"]
    assert validate_inventory_growth(
        before_inventory, after_inventory, domain="d01", ticks=900
    ) == after_inventory


def test_current_restart_rejects_missing_canonical_kf_phase_member(
        monkeypatch, tmp_path):
    """A v5 archive cannot turn eager held QS back into lazy state."""
    from gpuwm.core.kf import KainFritsch

    cfg = _cfg(moist=True, mp_physics=10, bl_pbl_physics=1,
               cu_physics=1, bldt=0.0)
    source, source_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    source_driver.cumulus_callable = KainFritsch()
    path = restart.write_restart(tmp_path / "canonical.npz", source, cfg)

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))
    missing = "driver/cumulus_tendencies/rqs"
    del payload[missing]
    del header["array_manifest"][missing]
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    tampered = tmp_path / "missing-rqs.npz"
    with tampered.open("wb") as stream:
        np.savez(stream, **payload)

    live, live_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(live)
    live_driver.cumulus_callable = KainFritsch()
    with pytest.raises(restart.RestartMismatchError,
                       match="missing canonical member.*rqs"):
        restart.restore_restart(tampered, live, cfg)


def test_restart_refuses_unfinalized_kf_expiry_transition(monkeypatch):
    """A checkpoint cannot retain an unconsumed transient expiry mask."""
    cfg = _cfg(moist=True, mp_physics=10, cu_physics=1)
    _state, driver = _shim_driver_state(cfg, monkeypatch)

    # The host recovery receipt alone is sufficient to reject a checkpoint:
    # an exception may have occurred before a nonzero mask was published.
    driver._cu_expiry_pending = True
    with pytest.raises(restart.RestartManifestError,
                       match="KF expiry finalization is pending"):
        restart._driver_manifest(driver)
    driver.finish_step()
    assert driver._cu_expiry_pending is False
    restart._driver_manifest(driver)

    # Keep probing the device mask for direct callers and synthetic states
    # that do not update the host receipt.
    driver.cu_expiring[0, 0] = np.float32(1.0)

    with pytest.raises(restart.RestartManifestError,
                       match="KF expiry finalization is pending"):
        restart._driver_manifest(driver)

    driver.finish_step()
    restart._driver_manifest(driver)


def test_restart_v5_serializes_one_microphysics_set_and_reads_bound_v2_layout(
        monkeypatch, tmp_path):
    import gpuwm.core.physics as physics

    cfg = _cfg(moist=True, mp_physics=10)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    for index, (component, slot) in enumerate(
            physics.microphysics_scratch_slots(cfg.mp_physics), start=1):
        state._scratch[slot][...] = np.float32(index) / np.float32(8.0)
        assert getattr(driver.microphysics, component) is state._scratch[slot]

    path = restart.write_restart(tmp_path / "v5.npz", state, cfg)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))
    assert header["format_version"] == 5
    assert not any(key.startswith("driver/microphysics/")
                   for key in header["array_manifest"])

    fresh, fresh_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    for component, slot in physics.microphysics_scratch_slots(cfg.mp_physics):
        assert getattr(fresh_driver.microphysics, component) is \
            fresh._scratch[slot]
        assert fresh._scratch[slot].tobytes() == state._scratch[slot].tobytes()

    # Synthesize the exact v2 dual-key layout from the v5 payload.  The array
    # shim remains testable only with the new mandatory physics-identity
    # header; genuinely old, unbound v2 files are intentionally refused.
    # It accepts the dual layout only while both historical copies have
    # identical bytes.
    header["format_version"] = 2
    for component, slot in physics.microphysics_scratch_slots(cfg.mp_physics):
        old_key = f"driver/microphysics/{component}"
        source = payload[f"scratch/{slot}"].copy()
        payload[old_key] = source
        header["array_manifest"][old_key] = {
            "shape": list(source.shape), "dtype": str(source.dtype)}
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    v2 = tmp_path / "v2.npz"
    with v2.open("wb") as stream:
        np.savez(stream, **payload)

    legacy, legacy_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(legacy)
    restart.restore_restart(v2, legacy, cfg)
    for component, slot in physics.microphysics_scratch_slots(cfg.mp_physics):
        assert getattr(legacy_driver.microphysics, component) is \
            legacy._scratch[slot]

    payload["driver/microphysics/rainnc"] = \
        payload["driver/microphysics/rainnc"].copy()
    payload["driver/microphysics/rainnc"].flat[0] += np.float32(1.0)
    mismatch = tmp_path / "v2-mismatch.npz"
    with mismatch.open("wb") as stream:
        np.savez(stream, **payload)
    rejected, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(rejected)
    with pytest.raises(restart.RestartMismatchError,
                       match="differ byte-for-byte"):
        restart.restore_restart(mismatch, rejected, cfg)


def test_wsm6_sr_exact_upper_roundtrips_restart_bits(monkeypatch, tmp_path):
    """The rebuilt guard metadata must not block or alter canonical SR."""
    from gpuwm.core.microphysics import MicrophysicsDiagnostics

    cfg = _cfg(moist=True, mp_physics=6, dt=60.0)
    source, source_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    shape = source.mup.shape
    upper = np.float32(source_driver._sr_roundoff_upper)
    source_driver.accept_microphysics(MicrophysicsDiagnostics(
        rainnc=np.ones(shape, np.float32),
        rainncv=np.ones(shape, np.float32),
        sr=np.full(shape, upper, np.float32)))

    path = restart.write_restart(tmp_path / "wsm6-upper.npz", source, cfg)
    fresh, fresh_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)

    assert fresh_driver._sr_roundoff_max_ulps == 3
    assert fresh_driver._wsm6_minor_loops == 1
    assert fresh_driver.microphysics.sr is fresh._scratch["mp_sr"]
    np.testing.assert_array_equal(
        fresh_driver.microphysics.sr.view(np.uint32),
        np.full(shape, upper, np.float32).view(np.uint32))


def test_synthetic_state_roundtrips_bit_exactly(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=10)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, seed=20260716)
    state.elapsed_seconds = 1234.5
    rng = np.random.default_rng(7)
    state.scratch((cfg.ny, cfg.nx), "mp_rainnc")[...] = \
        rng.standard_normal((cfg.ny, cfg.nx)).astype(np.float32)

    path = restart.write_restart(
        tmp_path / "rst.npz", state, cfg,
        run_trackers={"nan_free": True, "w_max_ms": 3.5})

    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)                       # identical deterministic setup
    info = restart.restore_restart(path, fresh, cfg)

    assert info.elapsed_seconds == 1234.5
    assert fresh.elapsed_seconds == 1234.5
    assert info.run_trackers == {"nan_free": True, "w_max_ms": 3.5}
    for name in restart.STATE_SERIALIZED_ATTRS:
        source = getattr(state, name, None)
        if source is None:
            continue
        target = getattr(fresh, name)
        assert target.dtype == source.dtype, name
        # Byte-level identity: NaN payloads, signed zeros, denormals.
        assert target.tobytes() == source.tobytes(), name
    assert (fresh._scratch["mp_rainnc"].tobytes()
            == state._scratch["mp_rainnc"].tobytes())

    header = restart.read_restart_header(path)
    assert header["format_version"] == restart.RESTART_FORMAT_VERSION
    assert header["elapsed_seconds"] == 1234.5
    assert header["config"]["nx"] == cfg.nx
    assert set(header["array_manifest"]) >= {"state/u", "state/h_diabatic",
                                             "scratch/mp_rainnc"}


def test_restore_normalizes_spec_zone_ring_microphysics(monkeypatch,
                                                        tmp_path):
    """v5 migration pin: a checkpoint seeded with nonzero spec-zone ring
    MP accumulators/*NCV/SR/refl and ring h_diabatic (producible only by
    the pre-ring-exclusion whole-field microphysics; no WRF-valid
    trajectory contains them -- allocator zero init
    frame/module_domain.F:770-777 + clipped tiles solve_em.F:3631-3639)
    restores with those rings zeroed while every interior byte
    round-trips exactly.  The same seeding under a periodic config
    restores byte-identically: the normalization is
    specified/nested-scoped.  Without it the ring guard's
    capture/restore would carry the stale ring forever, and ring
    h_diabatic would feed every RK stage.  The boundary-forced arm uses
    the nested-child flag (where d02/d03 ring precipitation actually
    accumulated pre-fix; the normalization path is identical for
    specified, but restoring a specified root requires the full
    attach_lateral_boundaries preparation this shim harness omits)."""
    surface_slots = ("mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
                     "mp_graupelnc", "mp_graupelncv", "mp_sr")
    for boundary_forced in (True, False):
        cfg = _cfg(nx=8, ny=7, moist=True, mp_physics=10,
                   nested=boundary_forced)
        state = _shim_state(cfg, monkeypatch)
        _fill_setup(state)
        _fill_serialized(state, seed=20260727)
        rng = np.random.default_rng(31)
        for slot in surface_slots:
            state.scratch((cfg.ny, cfg.nx), slot)[...] = 0.5 + np.abs(
                rng.standard_normal((cfg.ny, cfg.nx))).astype(np.float32)
        state.scratch((cfg.nz, cfg.ny, cfg.nx), "refl_10cm")[...] = (
            rng.standard_normal(
                (cfg.nz, cfg.ny, cfg.nx)).astype(np.float32))
        hd = rng.standard_normal(
            (cfg.nz, cfg.ny, cfg.nx)).astype(np.float32)
        state.h_diabatic[...] = hd
        path = restart.write_restart(
            tmp_path / f"ring-migration-{boundary_forced}.npz", state, cfg)

        fresh = _shim_state(cfg, monkeypatch)
        _fill_setup(fresh)
        restart.restore_restart(path, fresh, cfg)

        sz = cfg.spec_zone
        ring = np.ones((cfg.ny, cfg.nx), dtype=bool)
        ring[sz:cfg.ny - sz, sz:cfg.nx - sz] = False
        for slot in surface_slots:
            got = fresh._scratch[slot]
            src = state._scratch[slot]
            np.testing.assert_array_equal(got[~ring], src[~ring],
                                          err_msg=slot)
            if boundary_forced:
                assert (got[ring] == 0.0).all(), slot
            else:
                np.testing.assert_array_equal(got, src, err_msg=slot)
        # refl_10cm is rebuild-classified (REBUILT_SCRATCH_PREFIXES
        # "refl_"): checkpoints never serialize the stash, so it cannot
        # carry pre-fix ring bytes across a restart -- the restored state
        # has no refl_10cm slot at all.  h_diabatic IS serialized and is
        # the dangerous carrier (it feeds every RK stage).
        assert "refl_10cm" not in fresh._scratch
        got, src = fresh.h_diabatic, hd
        np.testing.assert_array_equal(got[:, ~ring], src[:, ~ring],
                                      err_msg="h_diabatic")
        if boundary_forced:
            assert (got[:, ring] == 0.0).all(), "h_diabatic"
        else:
            np.testing.assert_array_equal(got, src, err_msg="h_diabatic")


def _nssl2_restart_fixture(monkeypatch, *, seed: int = 1):
    cfg = _cfg(moist=True, mp_physics=18)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    for index, name in enumerate((
            *restart.NSSL2_RESTART_PROGNOSTICS,
            *restart.NSSL2_RESTART_AUXILIARY_STATE), start=1):
        value = np.float32(seed * 0.01 + index * 1.0e-4)
        getattr(state, name)[...] = value
    for index, slot in enumerate(
            restart.NSSL2_RESTART_PRECIPITATION_SLOTS, start=1):
        state._scratch[slot][...] = np.float32(seed + index * 0.125)
    driver._pending_rainbl[...] = np.float32(seed * 0.25)
    driver.microphysics_updates = 0
    state.elapsed_seconds = 0.0
    return cfg, state, driver


def _advance_nssl2_restart_fixture(state, driver, cfg, steps: int) -> None:
    """Deterministic FP32 continuation seam for restart inventory tests."""
    for _ in range(steps):
        update = driver.microphysics_updates
        first_call = update == 0
        scale = np.float32(update + 1)
        for index, name in enumerate(
                restart.NSSL2_RESTART_PROGNOSTICS, start=1):
            increment = (
                np.float32(index) * np.float32(2.5e-6) * scale)
            getattr(state, name)[...] += increment
        if first_call:
            # Mirrors the one-time calcnfromq gate's dependence on persistent
            # first-call state: repeating this after restore must diverge.
            state.qnn[...] += np.float32(17.0)
        state.h_diabatic[...] = (
            state.qv * np.float32(0.125)
            + np.float32(update) * np.float32(1.0e-5))
        for index, slot in enumerate(
                restart.NSSL2_RESTART_PRECIPITATION_SLOTS, start=1):
            state._scratch[slot][...] += (
                np.float32(index) * np.float32(0.03125) * scale)
        driver._pending_rainbl[...] += \
            state._scratch["mp_rainncv"] * np.float32(0.01)
        driver.microphysics_updates += 1
        state.elapsed_seconds += float(cfg.dt)


def _nssl2_restart_bytes(state) -> dict[str, bytes]:
    manifest = {}
    manifest.update(restart.state_manifest(state))
    manifest.update(restart._scratch_manifest(state))
    manifest.update(restart._driver_manifest(state.physics))
    return {
        key: restart._host(value).tobytes()
        for key, value in sorted(manifest.items())
    }


def test_nssl2_restart_contract_pins_exact_canonical_inventory(
        monkeypatch, tmp_path):
    cfg, state, driver = _nssl2_restart_fixture(monkeypatch)
    _advance_nssl2_restart_fixture(state, driver, cfg, 2)
    path = restart.write_restart(tmp_path / "nssl2.npz", state, cfg)
    header = restart.read_restart_header(path)

    assert header["format_version"] == 5
    contract = header["physics_setup"]["microphysics"]["restart_contract"]
    assert contract == {
        "schema_version": restart.NSSL2_RESTART_CONTRACT_VERSION,
        "physics_contract_id": NSSL2_CONTRACT_ID,
        "wrf_reference": {
            "version": NSSL2_WRF_REFERENCE_VERSION,
            "commit": NSSL2_WRF_REFERENCE_COMMIT,
        },
        "resolved_default_mode": asdict(NSSL2_DEFAULT_MODE),
        "resolved_wrf_namelist_defaults": dict(NSSL2_WRF_NAMELIST_DEFAULTS),
        "state_members": [
            *(f"state/{name}" for name in
              restart.NSSL2_RESTART_PROGNOSTICS),
            *(f"state/{name}" for name in
              restart.NSSL2_RESTART_AUXILIARY_STATE),
        ],
        "precipitation_members": [
            f"scratch/{slot}"
            for slot in restart.NSSL2_RESTART_PRECIPITATION_SLOTS
        ],
        "first_call_authority": "driver.microphysics_updates == 0",
        "clock_authority": "header.elapsed_seconds",
        "continuation_policy": "bitwise",
    }
    manifest = set(header["array_manifest"])
    assert set(contract["state_members"]) <= manifest
    assert set(contract["precipitation_members"]) <= manifest
    assert not (manifest & restart.NSSL2_LEGACY_RESTART_ALIASES)
    assert header["driver"]["microphysics_updates"] == 2
    assert header["elapsed_seconds"] == 2 * cfg.dt


def test_nssl2_split_run_restart_continuation_is_bitwise(
        monkeypatch, tmp_path):
    cfg, straight, straight_driver = _nssl2_restart_fixture(
        monkeypatch, seed=7)
    _advance_nssl2_restart_fixture(straight, straight_driver, cfg, 8)

    _, split, split_driver = _nssl2_restart_fixture(monkeypatch, seed=7)
    _advance_nssl2_restart_fixture(split, split_driver, cfg, 3)
    boundary_bytes = _nssl2_restart_bytes(split)
    path = restart.write_restart(tmp_path / "split.npz", split, cfg)
    header = restart.read_restart_header(path)
    assert not any("nssl2_driver_" in name
                   or "nssl2_fused_" in name
                   or "nssl2_primary_ice_" in name
                   or "nssl2_nucond_" in name
                   for name in header["array_manifest"])

    _, resumed, resumed_driver = _nssl2_restart_fixture(
        monkeypatch, seed=99)
    rebuilt_binding = resumed_driver.nssl2_binding
    rebuilt_ids = tuple(id(value) for value in (
        rebuilt_binding.workspace.state,
        rebuilt_binding.workspace.category_surface_export,
        rebuilt_binding.workspace.ignored_accumulator,
        rebuilt_binding.fused_gs.temperature_k,
        rebuilt_binding.fused_gs.primary_ice_target_m3,
        rebuilt_binding.nucond_scratch,
    ))
    assert rebuilt_binding is not split_driver.nssl2_binding
    restart.restore_restart(path, resumed, cfg)
    assert resumed_driver.nssl2_binding is rebuilt_binding
    assert tuple(id(value) for value in (
        rebuilt_binding.workspace.state,
        rebuilt_binding.workspace.category_surface_export,
        rebuilt_binding.workspace.ignored_accumulator,
        rebuilt_binding.fused_gs.temperature_k,
        rebuilt_binding.fused_gs.primary_ice_target_m3,
        rebuilt_binding.nucond_scratch,
    )) == rebuilt_ids
    rebuilt_binding.validate(resumed, cfg.dt)
    assert _nssl2_restart_bytes(resumed) == boundary_bytes
    assert resumed_driver.microphysics_updates == 3
    assert resumed.elapsed_seconds == 3 * cfg.dt

    _advance_nssl2_restart_fixture(resumed, resumed_driver, cfg, 5)
    assert _nssl2_restart_bytes(resumed) == _nssl2_restart_bytes(straight)
    for name in (
            *restart.NSSL2_RESTART_PROGNOSTICS,
            *restart.NSSL2_RESTART_AUXILIARY_STATE):
        assert getattr(resumed, name).tobytes() == \
            getattr(straight, name).tobytes(), name
    for slot in restart.NSSL2_RESTART_PRECIPITATION_SLOTS:
        assert resumed._scratch[slot].tobytes() == \
            straight._scratch[slot].tobytes(), slot
    assert resumed_driver.microphysics_updates == \
        straight_driver.microphysics_updates == 8
    assert resumed.elapsed_seconds == straight.elapsed_seconds == 8 * cfg.dt


@pytest.mark.parametrize(("key", "corrupt"), [
    ("state/qnr", "shape"),
    ("state/qvolh", "dtype"),
    ("scratch/mp_hailncv", "shape"),
    ("scratch/mp_sr", "dtype"),
])
def test_nssl2_restore_rejects_shape_and_dtype_before_mutation(
        monkeypatch, tmp_path, key, corrupt):
    cfg, source, _ = _nssl2_restart_fixture(monkeypatch)
    path = restart.write_restart(tmp_path / "source.npz", source, cfg)
    with np.load(path, allow_pickle=False) as data:
        original = data[key]
    replacement = (original.reshape(-1)[:-1]
                   if corrupt == "shape" else original.astype(np.float64))
    tampered = _replace_restart_member(
        path, tmp_path / f"{key.replace('/', '-')}-{corrupt}.npz",
        key, replacement)

    _, live, live_driver = _nssl2_restart_fixture(monkeypatch, seed=91)
    before = _nssl2_restart_bytes(live)
    with pytest.raises(restart.RestartMismatchError,
                       match=f"{key}.*(shape|dtype)"):
        restart.restore_restart(tampered, live, cfg)
    assert _nssl2_restart_bytes(live) == before
    assert live_driver.microphysics_updates == 0


@pytest.mark.parametrize("edit_name", [
    "missing", "extra", "generic-alias", "fortran-alias", "driver-alias",
])
def test_nssl2_restore_rejects_missing_extra_and_legacy_names_atomically(
        monkeypatch, tmp_path, edit_name):
    cfg, source, _ = _nssl2_restart_fixture(monkeypatch)
    path = restart.write_restart(tmp_path / "source.npz", source, cfg)

    def edit(payload, header):
        if edit_name == "missing":
            payload.pop("state/qvolh")
        elif edit_name == "extra":
            payload["state/qnr_legacy"] = payload["state/qnr"].copy()
        elif edit_name == "generic-alias":
            payload["state/nc"] = payload.pop("state/qndrop")
        elif edit_name == "fortran-alias":
            payload["state/ccw"] = payload.pop("state/qndrop")
        else:
            payload["driver/microphysics/rainnc"] = \
                payload.pop("scratch/mp_rainnc")

    tampered = _rewrite_restart_archive(
        path, tmp_path / f"{edit_name}.npz", edit)
    _, live, live_driver = _nssl2_restart_fixture(monkeypatch, seed=92)
    before = _nssl2_restart_bytes(live)
    match = "legacy" if "alias" in edit_name else "inventory"
    with pytest.raises(restart.RestartMismatchError, match=match):
        restart.restore_restart(tampered, live, cfg)
    assert _nssl2_restart_bytes(live) == before
    assert live_driver.microphysics_updates == 0


@pytest.mark.parametrize("edit_name", [
    "missing-contract", "wrong-contract-version", "extended-contract",
    "wrong-outer-version",
])
def test_nssl2_restore_rejects_unknown_contract_and_outer_versions(
        monkeypatch, tmp_path, edit_name):
    cfg, source, _ = _nssl2_restart_fixture(monkeypatch)
    path = restart.write_restart(tmp_path / "source.npz", source, cfg)

    def edit(payload, header):
        if edit_name == "wrong-outer-version":
            header["format_version"] = 4
            return
        microphysics = header["physics_setup"]["microphysics"]
        if edit_name == "missing-contract":
            microphysics.pop("restart_contract")
        elif edit_name == "wrong-contract-version":
            microphysics["restart_contract"]["schema_version"] = 0
        else:
            microphysics["restart_contract"]["legacy_permitted"] = True
        header["physics_setup_fingerprint"] = restart._json_sha256(
            header["physics_setup"])

    tampered = _rewrite_restart_archive(
        path, tmp_path / f"{edit_name}.npz", edit)
    _, live, live_driver = _nssl2_restart_fixture(monkeypatch, seed=93)
    before = _nssl2_restart_bytes(live)
    match = ("format version" if edit_name == "wrong-outer-version"
             else "MP18 restart contract")
    with pytest.raises(restart.RestartMismatchError, match=match):
        restart.restore_restart(tampered, live, cfg)
    assert _nssl2_restart_bytes(live) == before
    assert live_driver.microphysics_updates == 0


@pytest.mark.parametrize("corrupt", [
    "missing-driver", "missing-slot", "wrong-dtype", "bad-counter",
])
def test_nssl2_write_refuses_incomplete_live_state(
        monkeypatch, tmp_path, corrupt):
    cfg, state, driver = _nssl2_restart_fixture(monkeypatch)
    if corrupt == "missing-driver":
        state.physics = None
    elif corrupt == "missing-slot":
        state._scratch.pop("mp_hailncv")
    elif corrupt == "wrong-dtype":
        state.qnh = state.qnh.astype(np.float64)
    else:
        driver.microphysics_updates = -1

    with pytest.raises(restart.RestartManifestError, match="MP18 restart"):
        restart.write_restart(tmp_path / f"{corrupt}.npz", state, cfg)


def test_restore_rejects_config_mismatch(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    path = restart.write_restart(tmp_path / "rst.npz", state, cfg)

    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    with pytest.raises(restart.RestartMismatchError, match="dt"):
        restart.restore_restart(path, fresh, replace(cfg, dt=30.0))
    with pytest.raises(restart.RestartMismatchError, match="mp_physics"):
        restart.restore_restart(path, fresh, replace(cfg, mp_physics=10))
    # Run-length knobs are the sanctioned difference between the writing
    # and the resuming run.
    restart.restore_restart(
        path, fresh, replace(cfg, run_seconds=999.0, output_interval_s=50.0,
                             restart_interval_s=100.0))


def test_restore_rejects_setup_fingerprint_mismatch(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    path = restart.write_restart(tmp_path / "rst.npz", state, cfg)

    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    fresh.thb[0] += np.float32(0.5)          # a different base state
    with pytest.raises(restart.RestartMismatchError, match="setup"):
        restart.restore_restart(path, fresh, cfg)


def test_thompson_restart_identity_binds_implementation_and_table_bytes(
        monkeypatch, tmp_path):
    """Experimental mp8 cannot cross a table or implementation boundary."""
    cfg = _cfg(moist=True, mp_physics=8)
    monkeypatch.setenv("GPUWM_EXPERIMENTAL_THOMPSON_MP8", "1")
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", "/fixture/thompson")
    generation = {"sha256": "a" * 64}

    def table_identity(path):
        assert path == "/fixture/thompson"
        return {
            "schema": 1,
            "table_set": "fixture-classic-thompson",
            "wrf_version": "v4.6.1",
            "wrf_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
            "assets": [{
                "filename": "fixture.dat", "bytes": 16,
                "sha256": generation["sha256"],
            }],
        }

    monkeypatch.setattr(restart, "_thompson_table_identity", table_identity)
    source, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    identity = restart.physics_setup_identity(source, cfg)
    assert identity["algorithms"]["microphysics"] == \
        restart.MICROPHYSICS_ALGORITHM_IDENTITIES[8]
    thompson = identity["microphysics"]["thompson"]
    assert thompson["tables"]["assets"][0]["sha256"] == "a" * 64
    assert thompson["graupel_number_policy"].startswith("wrf-private")
    assert thompson["reflectivity_policy"].endswith("output-only-v1")
    assert "png-scw-held-number" in thompson["snow_rime_conversion_policy"]
    assert "vts-boost" in thompson["snow_fall_speed_policy"]

    path = restart.write_restart(tmp_path / "thompson.npz", source, cfg)
    same, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(same)
    restart.restore_restart(path, same, cfg)

    generation["sha256"] = "b" * 64
    changed, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(changed)
    before = changed.qg.tobytes()
    with pytest.raises(restart.RestartMismatchError,
                       match="physics setup"):
        restart.restore_restart(path, changed, cfg)
    assert changed.qg.tobytes() == before


def test_thompson_restart_identity_resolves_packaged_root_without_env(
        monkeypatch):
    """The identity binds the RESOLVED root: packaged default, env override.

    Until the mp8 promotion (product/v1 packaging lane 2026-07-28) this
    test asserted that identity construction refused to run without
    GPUWM_EXPERIMENTAL_THOMPSON_MP8=1 and GPUWM_THOMPSON_TABLE_ROOT.  The
    guard is retired and the table root defaults to the packaged
    gpuwm/data/thompson/tables directory; what this test now pins is that
    the restart identity resolves exactly the root the forecast adapter
    loads, records the promoted admission token instead of the retired
    guard, and still honors the override.
    """
    from gpuwm.physics_compat import packaged_thompson_table_root

    cfg = _cfg(moist=True, mp_physics=8)
    state, _ = _shim_driver_state(cfg, monkeypatch)
    seen = []

    def table_identity(path):
        seen.append(path)
        return {"schema": 1, "table_set": "fixture", "assets": []}

    monkeypatch.setattr(restart, "_thompson_table_identity", table_identity)

    monkeypatch.delenv("GPUWM_EXPERIMENTAL_THOMPSON_MP8", raising=False)
    monkeypatch.delenv("GPUWM_THOMPSON_TABLE_ROOT", raising=False)
    identity = restart.physics_setup_identity(state, cfg)
    thompson = identity["microphysics"]["thompson"]
    assert seen == [str(packaged_thompson_table_root())]
    assert thompson["admission"] == "first-class-mp8-packaged-tables-v1"
    assert "implementation_guard" not in thompson

    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", "/override/thompson")
    restart.physics_setup_identity(state, cfg)
    assert seen[-1] == "/override/thompson"


def test_restart_header_binds_resolved_physics_and_active_assets(
        monkeypatch, tmp_path):
    """The identity is an auditable resolved setup, not only scheme IDs.

    In particular, ra_physics=4 must say which radiation algorithm and
    above-atmosphere policy produced the held heating/flux state, and must
    bind every packaged coefficient/climatology file it actually consumes.
    """
    cfg = _cfg(
        moist=True, mp_physics=10, morr_rimed_ice=1,
        sf_sfclay_physics=1, sf_surface_physics=2, bl_pbl_physics=1,
        ra_physics=4, cu_physics=1,
    )
    state, driver = _identity_bound_physics_state(cfg, monkeypatch)
    monkeypatch.setattr(
        restart, "_asset_sha256",
        lambda path: f"test-sha256:{Path(path).name}")

    path = restart.write_restart(tmp_path / "physics-identity.npz", state,
                                 cfg)
    header = restart.read_restart_header(path)
    identity = header["physics_setup"]

    assert header["format_version"] == 5
    assert header["physics_setup_fingerprint"] == \
        restart.physics_setup_fingerprint(state, cfg)
    assert identity["schema_version"] == restart.PHYSICS_SETUP_SCHEMA_VERSION
    assert identity["algorithms"]["radiation"] == \
        restart.RADIATION_ALGORITHM_IDENTITIES[4]
    radiation = identity["radiation"]
    assert radiation["algorithm"] == restart.RADIATION_ALGORITHM_IDENTITIES[4]
    assert radiation["above_atmosphere_policy"] == \
        restart.RADIATION_ABOVE_ATMOSPHERE_POLICIES[4]
    assert radiation["start_time"] == "1974-04-03T12:00:00"
    assert radiation["trace_vmr"] == {"co2": 3.30e-4, "n2o": 3.0e-7}
    assert radiation["trace_gas_overrides"] == {"co2": 3.30e-4}
    assert radiation["latitude"]["shape"] == [cfg.ny, cfg.nx]
    assert radiation["ozone_vmr"]["dtype"] == "float32"

    assert set(identity["assets"]) == {
        "rrtmgp_gas_lw", "rrtmgp_gas_sw",
        "rrtmgp_cloud_lw", "rrtmgp_cloud_sw", "rrtmgp_rfmip",
        "noah_vegparm", "noah_soilparm", "noah_genparm", "noah_landuse",
        "kf_lutab",
    }
    for asset in identity["assets"].values():
        assert asset["path"].startswith("data/")
        assert asset["sha256"] == \
            f"test-sha256:{Path(asset['path']).name}"
    assert identity["land_surface"]["parameters"]["arrays"]["veg"][
        "shape"] == [2, 15]
    assert identity["microphysics"]["morrison_rimed_ice"] == {
        "selection": 1, "ag": 114.5, "bg": 0.5,
        "rhog": 900.0, "cg": pytest.approx(150.0 * np.pi),
    }

    fresh, _ = _identity_bound_physics_state(cfg, monkeypatch)
    restart.restore_restart(path, fresh, cfg)

    # A diagnostic call counter does not define setup and must not make a
    # freshly constructed adapter incompatible with its own restart.
    before = restart.physics_setup_fingerprint(state, cfg)
    driver.radiation_callable.update_count = 123
    assert restart.physics_setup_fingerprint(state, cfg) == before


def test_stock_analytic_radiation_identity_roundtrips_without_gpu(
        monkeypatch, tmp_path):
    from gpuwm.core import analytic_radiation

    cfg = _cfg(moist=True, ra_physics=90)
    monkeypatch.setattr(analytic_radiation, "cp", _NumpyCupyShim)

    def prepared(latitude_offset=0.0):
        state, driver = _shim_driver_state(cfg, monkeypatch)
        _fill_setup(state)
        shape = state.mup.shape
        driver.radiation_callable = \
            analytic_radiation.AnalyticClearSkyRadiation(
                datetime(1974, 4, 3, 12),
                np.full(shape, 39.0 + latitude_offset, dtype=np.float32),
                np.full(shape, -87.0, dtype=np.float32))
        return state

    source = prepared()
    path = restart.write_restart(tmp_path / "analytic.npz", source, cfg)
    header = restart.read_restart_header(path)
    radiation = header["physics_setup"]["radiation"]
    assert radiation["callable"] == {
        "class": "gpuwm.core.analytic_radiation.AnalyticClearSkyRadiation",
        "implementation": "stock",
    }
    assert radiation["above_atmosphere_policy"] == \
        restart.RADIATION_ABOVE_ATMOSPHERE_POLICIES[90]
    assert radiation["constants"]["clear_sky_transmissivity"] == 0.75
    assert header["physics_setup"]["assets"] == {}

    restart.restore_restart(path, prepared(), cfg)
    with pytest.raises(restart.RestartMismatchError,
                       match="physics setup"):
        restart.restore_restart(path, prepared(latitude_offset=0.5), cfg)


def test_active_surface_identity_binds_packaged_landuse_coefficients(
        monkeypatch):
    cfg = _cfg(
        moist=True, sf_sfclay_physics=1, sf_surface_physics=2,
        bl_pbl_physics=1,
    )
    state, _ = _identity_bound_physics_state(cfg, monkeypatch)
    monkeypatch.setattr(
        restart, "_asset_sha256",
        lambda path: f"test-sha256:{Path(path).name}")

    landuse = restart.physics_setup_identity(state, cfg)["assets"][
        "noah_landuse"]

    assert landuse == {
        "path": "data/noah_tables/LANDUSE.TBL",
        "bytes": 41236,
        "sha256": "test-sha256:LANDUSE.TBL",
    }


def test_stock_rrtmgp_and_kf_identities_bind_live_resolved_objects(
        monkeypatch):
    from gpuwm.core import kf

    cfg = _cfg(
        moist=True, mp_physics=10, sf_sfclay_physics=1,
        sf_surface_physics=2, bl_pbl_physics=1,
        ra_physics=4, cu_physics=1,
    )
    state, driver = _identity_bound_physics_state(cfg, monkeypatch)
    monkeypatch.setattr(restart, "_asset_sha256", lambda path: "disk-bytes")
    table = kf.KFTable(
        temperature=np.array([[1.0, 2.0]], dtype=np.float32),
        qsat=np.array([[3.0, 4.0]], dtype=np.float32),
        thetae_base=np.array([5.0], dtype=np.float32),
        log_ratio=np.array([6.0], dtype=np.float32),
        pressure_top=10000.0, pressure_reciprocal=1.0,
        thetae_reciprocal=2.0,
    )
    monkeypatch.setattr(kf, "load_kf_table", lambda: table)

    baseline = restart.physics_setup_fingerprint(state, cfg)
    driver.radiation_callable.lw_tables.coefficient[0] += 0.25
    changed_rrtmgp = restart.physics_setup_fingerprint(state, cfg)
    assert changed_rrtmgp != baseline
    driver.radiation_callable.lw_tables.coefficient[0] -= 0.25

    driver.radiation_callable.chunk_workspace = SimpleNamespace(
        nz=cfg.nz, column_chunk=17, p_top=10000.0, nbytes=4096,
        _phase_layouts={"lw": {"tau": ((17, cfg.nz), 4)}},
    )
    changed_workspace = restart.physics_setup_fingerprint(state, cfg)
    assert changed_workspace != baseline
    driver.radiation_callable.chunk_workspace.p_top = 10001.0
    changed_workspace_top = restart.physics_setup_fingerprint(state, cfg)
    assert changed_workspace_top != changed_workspace
    driver.radiation_callable.chunk_workspace = None

    table.temperature[0, 0] += np.float32(0.5)
    changed_kf = restart.physics_setup_fingerprint(state, cfg)
    assert changed_kf != baseline


def test_declared_custom_physics_identity_does_not_require_stock_internals(
        monkeypatch, tmp_path):
    cfg = _cfg(moist=True, ra_physics=4, cu_physics=1)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    driver.radiation_callable = _DeclaredPhysicsCallable("custom-radiation")
    driver.cumulus_callable = _DeclaredPhysicsCallable("custom-cumulus")
    driver.cumulus_callable.w0avg = None

    identity = restart.physics_setup_identity(state, cfg)

    assert identity["radiation"]["callable"]["declared_identity"][
        "algorithm"] == "custom-radiation"
    assert identity["cumulus"]["callable"]["declared_identity"][
        "algorithm"] == "custom-cumulus"
    assert not set(identity["assets"]) & {
        "rrtmgp_gas_lw", "rrtmgp_gas_sw", "rrtmgp_cloud_lw",
        "rrtmgp_cloud_sw", "rrtmgp_rfmip", "kf_lutab",
    }

    path = restart.write_restart(tmp_path / "custom-physics.npz", state, cfg)
    fresh, fresh_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(fresh)
    fresh_driver.radiation_callable = \
        _DeclaredPhysicsCallable("custom-radiation")
    fresh_driver.cumulus_callable = _DeclaredPhysicsCallable("custom-cumulus")
    fresh_driver.cumulus_callable.w0avg = None
    restart.restore_restart(path, fresh, cfg)


@pytest.mark.parametrize(
    "mismatch",
    ["radiation-geometry", "resolved-trace-gas", "noah-parameters",
     "asset-bytes", "radiation-algorithm", "above-atmosphere-policy",
     "land-surface-algorithm", "resolved-driver-switch"],
)
def test_restore_rejects_changed_physics_setup_before_mutation(
        monkeypatch, tmp_path, mismatch):
    """Every trajectory-changing resolved setup input is a refusal gate."""
    cfg = _cfg(
        moist=True, mp_physics=10, morr_rimed_ice=1,
        sf_sfclay_physics=1, sf_surface_physics=2, bl_pbl_physics=1,
        ra_physics=4, cu_physics=1,
    )
    asset_generation = {"value": "writer"}
    monkeypatch.setattr(
        restart, "_asset_sha256",
        lambda path: f"{asset_generation['value']}:{Path(path).name}")
    source, _ = _identity_bound_physics_state(cfg, monkeypatch)
    path = restart.write_restart(tmp_path / f"{mismatch}.npz", source, cfg)

    kwargs = {}
    if mismatch == "radiation-geometry":
        kwargs["latitude_offset"] = 0.25
    elif mismatch == "resolved-trace-gas":
        kwargs["trace_co2"] = 3.31e-4
    elif mismatch == "noah-parameters":
        kwargs["noah_offset"] = 0.5
    live, live_driver = _identity_bound_physics_state(
        cfg, monkeypatch, **kwargs)
    _fill_serialized(live, seed=20260718)
    live.elapsed_seconds = 47.0
    before = {
        name: getattr(live, name).tobytes()
        for name in restart.STATE_SERIALIZED_ATTRS
        if getattr(live, name, None) is not None}

    if mismatch == "asset-bytes":
        asset_generation["value"] = "reader"
    elif mismatch == "radiation-algorithm":
        monkeypatch.setitem(
            restart.RADIATION_ALGORITHM_IDENTITIES, 4,
            "rte-rrtmgp-deliberately-different-test")
    elif mismatch == "above-atmosphere-policy":
        monkeypatch.setitem(
            restart.RADIATION_ABOVE_ATMOSPHERE_POLICIES, 4,
            "explicit-upper-atmosphere-test-policy")
    elif mismatch == "land-surface-algorithm":
        monkeypatch.setitem(
            restart.LAND_SURFACE_ALGORITHM_IDENTITIES, 2,
            "pre-chs2-and-source-water-lake-skin-test-identity")
    elif mismatch == "resolved-driver-switch":
        live_driver.surface_enabled = False

    with pytest.raises(restart.RestartMismatchError,
                       match="physics setup"):
        restart.restore_restart(path, live, cfg)

    assert live.elapsed_seconds == 47.0
    assert before == {
        name: getattr(live, name).tobytes() for name in before}


def test_restart_rejects_missing_or_self_inconsistent_physics_identity(
        monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    path = restart.write_restart(tmp_path / "source.npz", state, cfg)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))

    missing = dict(header)
    missing.pop("physics_setup_fingerprint")
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(missing).encode("utf-8"), dtype=np.uint8)
    missing_path = tmp_path / "missing-identity.npz"
    with missing_path.open("wb") as stream:
        np.savez(stream, **payload)
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    with pytest.raises(restart.RestartMismatchError,
                       match="physics_setup_fingerprint"):
        restart.restore_restart(missing_path, fresh, cfg)

    header["physics_setup"]["algorithms"]["microphysics"] = \
        "silently-tampered"
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    inconsistent = tmp_path / "inconsistent-identity.npz"
    with inconsistent.open("wb") as stream:
        np.savez(stream, **payload)
    other = _shim_state(cfg, monkeypatch)
    _fill_setup(other)
    with pytest.raises(restart.RestartMismatchError,
                       match="physics setup fingerprint"):
        restart.restore_restart(inconsistent, other, cfg)


def test_restore_rejects_live_w0avg_when_archive_has_no_history(
        monkeypatch, tmp_path):
    cfg = _cfg(moist=True)
    source, source_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    source_driver.cumulus_callable = SimpleNamespace(w0avg=None)
    path = restart.write_restart(tmp_path / "no-w0avg.npz", source, cfg)

    live, live_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(live)
    live_driver.cumulus_callable = SimpleNamespace(
        w0avg=np.full(live.p.shape, 17.0, dtype=np.float32))
    before = live.u.tobytes()
    with pytest.raises(restart.RestartMismatchError, match="W0AVG"):
        restart.restore_restart(path, live, cfg)
    assert live.u.tobytes() == before
    np.testing.assert_array_equal(live_driver.cumulus_callable.w0avg, 17.0)


@pytest.mark.parametrize(
    ("key", "corrupt"), [
        ("driver/pbl_tendencies/ru", "shape"),
        ("driver/pbl_tendencies/ru", "dtype"),
        ("cumulus/w0avg", "shape"),
        ("cumulus/w0avg", "dtype"),
    ],
    ids=["held-shape", "held-dtype", "w0avg-shape", "w0avg-dtype"],
)
def test_restore_prevalidates_held_and_w0avg_arrays_atomically(
        monkeypatch, tmp_path, key, corrupt):
    """All driver payload contracts are checked before one live byte moves."""
    cfg = _cfg(moist=True)
    source, source_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    _fill_serialized(source, seed=1)
    source_driver.cumulus_callable = SimpleNamespace(
        w0avg=np.zeros(source.mup.shape, dtype=np.float32))
    path = restart.write_restart(tmp_path / "source.npz", source, cfg)

    with np.load(path, allow_pickle=False) as data:
        original = data[key]
    replacement = (original.reshape(-1)[:-1]
                   if corrupt == "shape"
                   else original.astype(np.float64))
    tampered = _replace_restart_member(
        path, tmp_path / f"{key.replace('/', '-')}-{corrupt}.npz",
        key, replacement)

    live, live_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(live)
    _fill_serialized(live, seed=2)
    live_driver.cumulus_callable = SimpleNamespace(
        w0avg=np.full(live.mup.shape, 7.0, dtype=np.float32))
    before_state = {
        name: getattr(live, name).tobytes()
        for name in restart.STATE_SERIALIZED_ATTRS
        if getattr(live, name, None) is not None}
    before_driver = {
        f"{tend}/{component}": getattr(getattr(live_driver, tend), component).tobytes()
        for tend in restart.DRIVER_TENDENCY_ATTRS
        for component in restart.TENDENCY_REQUIRED_COMPONENTS}
    before_w0avg = live_driver.cumulus_callable.w0avg.tobytes()

    with pytest.raises(restart.RestartMismatchError,
                       match=f"{key}.*(shape|dtype)"):
        restart.restore_restart(tampered, live, cfg)

    assert before_state == {
        name: getattr(live, name).tobytes() for name in before_state}
    assert before_driver == {
        f"{tend}/{component}": getattr(getattr(live_driver, tend), component).tobytes()
        for tend in restart.DRIVER_TENDENCY_ATTRS
        for component in restart.TENDENCY_REQUIRED_COMPONENTS}
    assert live_driver.cumulus_callable.w0avg.tobytes() == before_w0avg


def test_restore_rejects_tampered_member_set(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    path = restart.write_restart(tmp_path / "rst.npz", state, cfg)

    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    removed = next(key for key in payload if key.startswith("state/"))
    payload.pop(removed)
    tampered = tmp_path / "tampered.npz"
    with tampered.open("wb") as stream:
        np.savez(stream, **payload)

    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    with pytest.raises(restart.RestartMismatchError, match="manifest"):
        restart.restore_restart(tampered, fresh, cfg)


def test_lbc_forcing_tables_are_pinned_by_the_fingerprint(monkeypatch,
                                                          tmp_path):
    """Review F3: a same-config resume against modified boundary forcing
    (a replaced/edited reference bundle) must be rejected — the setup
    fingerprint digests every interval's side tables byte-level."""
    from gpuwm.ingest.lateral_bc import build_lateral_boundaries

    cfg = _cfg(moist=True, mp_physics=1)

    def boundaries(seed):
        rng = np.random.default_rng(seed)
        snapshots = [{"u": rng.standard_normal((8, 8))} for _ in range(2)]
        return build_lateral_boundaries(
            snapshots, [0.0, 3600.0], spec_bdy_width=3, spec_zone=1,
            relax_zone=2)

    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    state.lateral_boundaries = boundaries(1)
    path = restart.write_restart(tmp_path / "rst.npz", state, cfg)

    # An identically rebuilt forcing set restores.
    same = _shim_state(cfg, monkeypatch)
    _fill_setup(same)
    same.lateral_boundaries = boundaries(1)
    restart.restore_restart(path, same, cfg)

    # Different forcing tables under the same config are refused.
    other = _shim_state(cfg, monkeypatch)
    _fill_setup(other)
    other.lateral_boundaries = boundaries(2)
    with pytest.raises(restart.RestartMismatchError, match="setup"):
        restart.restore_restart(path, other, cfg)

    # Forcing presence itself is fingerprinted: attached-vs-None differs.
    none_state = _shim_state(cfg, monkeypatch)
    _fill_setup(none_state)
    with pytest.raises(restart.RestartMismatchError, match="setup"):
        restart.restore_restart(path, none_state, cfg)


def _historical_setup_fingerprint(state) -> str:
    """The pre-extension digest algorithm, copied here as a freeze oracle."""
    digest = hashlib.sha256()

    def array(name, value):
        host = np.asarray(value)
        digest.update(name.encode())
        digest.update(str(host.shape).encode())
        digest.update(str(host.dtype).encode())
        digest.update(host.tobytes())

    for name in restart.STATE_SETUP_ARRAYS:
        array(name, getattr(state, name))
    for name in restart.STATE_SETUP_SCALARS:
        value = getattr(state, name)
        if value is not None and not isinstance(value, bool):
            value = float(value)
        digest.update(f"{name}={value!r};".encode())
    nest_class = getattr(state, "_nest_restart_classification", None)
    if nest_class is not None:
        assert nest_class == "REBUILT"
        digest.update(b"nest_tables=REBUILT;")
    else:
        boundaries = getattr(state, "lateral_boundaries", None)
        if boundaries is None:
            digest.update(b"lateral_boundaries=None;")
        else:
            digest.update(
                f"lbc:width={boundaries.spec_bdy_width};"
                f"spec={boundaries.spec_zone};"
                f"relax={boundaries.relax_zone};"
                f"intervals={len(boundaries.intervals)};".encode())
            for interval in boundaries.intervals:
                digest.update(
                    f"[{interval.start_seconds!r},"
                    f"{interval.end_seconds!r}]".encode())
                for name in sorted(interval.fields):
                    boundary = interval.fields[name]
                    for side_name in ("west", "east", "south", "north"):
                        side = getattr(boundary, side_name)
                        array(f"{name}/{side_name}/value", side.value)
                        array(f"{name}/{side_name}/tendency", side.tendency)
    return digest.hexdigest()


def _extension_boundaries(count: int, *, seed=11, seconds=None,
                          spec_bdy_width=3, spec_zone=1, relax_zone=2,
                          extra_field=False):
    from gpuwm.ingest.lateral_bc import build_lateral_boundaries

    rng = np.random.default_rng(seed)
    snapshots = []
    for _ in range(count):
        row = {"u": rng.standard_normal((8, 8))}
        if extra_field:
            row["v"] = rng.standard_normal((8, 8))
        snapshots.append(row)
    if seconds is None:
        seconds = [float(index * 3600) for index in range(count)]
    return build_lateral_boundaries(
        snapshots, seconds, spec_bdy_width=spec_bdy_width,
        spec_zone=spec_zone, relax_zone=relax_zone)


def _extension_state(cfg, monkeypatch, *, count: int, seed=11,
                     seconds=None, spec_bdy_width=3, spec_zone=1,
                     relax_zone=2, extra_field=False):
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    state.lateral_boundaries = _extension_boundaries(
        count, seed=seed, seconds=seconds,
        spec_bdy_width=spec_bdy_width, spec_zone=spec_zone,
        relax_zone=relax_zone, extra_field=extra_field)
    return state


def test_setup_fingerprint_refactor_preserves_legacy_bytes(monkeypatch):
    cfg = _cfg(moist=True, mp_physics=1)
    root = _extension_state(cfg, monkeypatch, count=3)
    assert restart.setup_fingerprint(root) == \
        _historical_setup_fingerprint(root)

    child = _shim_state(replace(cfg, grid_id=2, nested=True), monkeypatch)
    _fill_setup(child)
    child._nest_restart_classification = "REBUILT"
    assert restart.setup_fingerprint(child) == \
        _historical_setup_fingerprint(child)


def test_exact_restart_still_refuses_forcing_extension(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    source = _extension_state(cfg, monkeypatch, count=2)
    source.elapsed_seconds = 3600.0
    path = restart.write_restart(tmp_path / "exact.npz", source, cfg)
    live = _extension_state(cfg, monkeypatch, count=3)
    with pytest.raises(restart.RestartMismatchError, match="setup"):
        restart.restore_restart(path, live, cfg)


def test_sealed_extension_requires_writer_opt_in_and_a_suffix(
        monkeypatch, tmp_path):
    cfg = _cfg(
        moist=True, mp_physics=1, specified=True,
        spec_bdy_width=3, spec_zone=1, relax_zone=2)
    source = _extension_state(cfg, monkeypatch, count=2)
    source.elapsed_seconds = 3600.0
    ordinary = restart.write_restart(tmp_path / "ordinary.npz", source, cfg)
    extended = _extension_state(cfg, monkeypatch, count=3)
    with pytest.raises(restart.RestartMismatchError, match="not intentionally"):
        restart._validate_restart(
            ordinary, extended, cfg, sealed_forcing_extension=True)

    sealed = restart.write_restart(
        tmp_path / "sealed.npz", source, cfg,
        sealed_forcing_extension=True)
    same = _extension_state(cfg, monkeypatch, count=2)
    with pytest.raises(restart.RestartMismatchError,
                       match="append at least one"):
        restart._validate_restart(
            sealed, same, cfg, sealed_forcing_extension=True)


@pytest.mark.parametrize("mutation", [
    "old-bytes", "bounds", "controls", "fields", "gap",
])
def test_sealed_extension_rejects_any_non_append_change(
        monkeypatch, tmp_path, mutation):
    cfg = _cfg(
        moist=True, mp_physics=1, specified=True,
        spec_bdy_width=3, spec_zone=1, relax_zone=2)
    source = _extension_state(cfg, monkeypatch, count=2)
    source.elapsed_seconds = 3600.0
    path = restart.write_restart(
        tmp_path / f"sealed-{mutation}.npz", source, cfg,
        sealed_forcing_extension=True)
    kwargs = {}
    if mutation == "bounds":
        kwargs["seconds"] = [0.0, 3590.0, 7200.0]
    elif mutation == "controls":
        kwargs["spec_bdy_width"] = 4
        kwargs["relax_zone"] = 3
    elif mutation == "fields":
        kwargs["extra_field"] = True
    elif mutation == "old-bytes":
        kwargs["seed"] = 12
    live = _extension_state(cfg, monkeypatch, count=3, **kwargs)
    if mutation == "gap":
        original = restart.lateral_boundary_prefix_identity

        def identity(state):
            result = original(state)
            if state is live:
                result = json.loads(json.dumps(result))
                result["intervals"][1]["start_seconds"] = 3700.0
            return result

        monkeypatch.setattr(restart, "lateral_boundary_prefix_identity", identity)
    with pytest.raises(restart.RestartMismatchError,
                       match="forcing|boundary"):
        restart._validate_restart(
            path, live, cfg, sealed_forcing_extension=True)


def test_sealed_extension_rejects_changed_suffix_seam(
        monkeypatch, tmp_path):
    from gpuwm.ingest.lateral_bc import (
        BoundaryInterval, LateralBoundaries, SideBoundary,
    )

    cfg = _cfg(
        moist=True, mp_physics=1, specified=True,
        spec_bdy_width=3, spec_zone=1, relax_zone=2)
    source = _extension_state(cfg, monkeypatch, count=2)
    source.elapsed_seconds = 3600.0
    path = restart.write_restart(
        tmp_path / "sealed-seam.npz", source, cfg,
        sealed_forcing_extension=True)
    live = _extension_state(cfg, monkeypatch, count=3)
    intervals = list(live.lateral_boundaries.intervals)
    suffix = intervals[1]
    fields = dict(suffix.fields)
    boundary = fields["u"]
    changed = np.array(boundary.west.value, copy=True)
    changed.flat[0] += 123.0
    fields["u"] = replace(
        boundary,
        west=SideBoundary(changed, boundary.west.tendency))
    intervals[1] = BoundaryInterval(
        suffix.start_seconds, suffix.end_seconds, fields)
    live.lateral_boundaries = LateralBoundaries(
        tuple(intervals), live.lateral_boundaries.spec_bdy_width,
        live.lateral_boundaries.spec_zone,
        live.lateral_boundaries.relax_zone)

    with pytest.raises(restart.RestartMismatchError,
                       match="discontinuous live forcing frame"):
        restart._validate_restart(
            path, live, cfg, sealed_forcing_extension=True)


def test_sealed_extension_binds_checkpoint_endpoint_setup_config_and_physics(
        monkeypatch, tmp_path):
    cfg = _cfg(
        moist=True, mp_physics=1, specified=True,
        spec_bdy_width=3, spec_zone=1, relax_zone=2)
    source = _extension_state(cfg, monkeypatch, count=2)
    source.elapsed_seconds = 3500.0
    endpoint = tmp_path / "endpoint.npz"
    with pytest.raises(restart.RestartMismatchError, match="not sealed"):
        restart.write_restart(
            endpoint, source, cfg, sealed_forcing_extension=True)
    assert not endpoint.exists()

    source.elapsed_seconds = 3600.0
    path = restart.write_restart(
        tmp_path / "identity.npz", source, cfg,
        sealed_forcing_extension=True)
    live = _extension_state(cfg, monkeypatch, count=3)
    changed_setup = _extension_state(cfg, monkeypatch, count=3)
    changed_setup.thb.flat[0] += np.float32(0.5)
    with pytest.raises(restart.RestartMismatchError, match="immutable setup"):
        restart._validate_restart(
            path, changed_setup, cfg, sealed_forcing_extension=True)
    with pytest.raises(restart.RestartMismatchError, match="dt"):
        restart._validate_restart(
            path, live, replace(cfg, dt=cfg.dt * 2),
            sealed_forcing_extension=True)

    tampered = _rewrite_header(
        path, tmp_path / "physics.npz",
        lambda header: header.__setitem__(
            "physics_setup_fingerprint", "0" * 64))
    with pytest.raises(restart.RestartMismatchError, match="physics setup"):
        restart._validate_restart(
            tampered, live, cfg, sealed_forcing_extension=True)


def test_sealed_extension_header_tamper_and_rebuilt_child_contract(
        monkeypatch, tmp_path):
    cfg = _cfg(
        moist=True, mp_physics=1, specified=True,
        spec_bdy_width=3, spec_zone=1, relax_zone=2)
    source = _extension_state(cfg, monkeypatch, count=2)
    source.elapsed_seconds = 3600.0
    path = restart.write_restart(
        tmp_path / "root.npz", source, cfg,
        sealed_forcing_extension=True)

    def alter(header):
        header["lateral_boundary_prefix"]["intervals"][0]["sha256"] = \
            "0" * 64

    tampered = _rewrite_header(path, tmp_path / "tampered.npz", alter)
    live = _extension_state(cfg, monkeypatch, count=3)
    with pytest.raises(restart.RestartMismatchError, match="sealed interval"):
        restart._validate_restart(
            tampered, live, cfg, sealed_forcing_extension=True)

    changed_prefix = _extension_state(cfg, monkeypatch, count=3, seed=12)
    forged_setup = _rewrite_header(
        path, tmp_path / "forged-setup.npz",
        lambda header: header.__setitem__(
            "setup_fingerprint",
            restart.setup_fingerprint(changed_prefix)))
    with pytest.raises(restart.RestartMismatchError, match="sealed interval"):
        restart._validate_restart(
            forged_setup, changed_prefix, cfg,
            sealed_forcing_extension=True)

    extra_key = _rewrite_header(
        path, tmp_path / "extra-prefix-key.npz",
        lambda header: header["lateral_boundary_prefix"].__setitem__(
            "unrecognized", True))
    with pytest.raises(restart.RestartMismatchError,
                       match="malformed sealed forcing document"):
        restart._validate_restart(
            extra_key, live, cfg, sealed_forcing_extension=True)

    missing_root_prefix = _rewrite_header(
        path, tmp_path / "missing-root-prefix.npz",
        lambda header: header.__setitem__("lateral_boundary_prefix", None))
    before = live.u.copy()
    before_elapsed = live.elapsed_seconds
    with pytest.raises(restart.RestartMismatchError,
                       match="specified root has no sealed"):
        restart._validate_restart(
            missing_root_prefix, live, cfg,
            sealed_forcing_extension=True)
    assert np.array_equal(live.u, before, equal_nan=True)
    assert live.elapsed_seconds == before_elapsed

    child_cfg = replace(cfg, grid_id=2, specified=False, nested=True)
    child = _shim_state(child_cfg, monkeypatch)
    _fill_setup(child)
    child._nest_restart_classification = "REBUILT"
    child_path = restart.write_restart(
        tmp_path / "child.npz", child, child_cfg,
        sealed_forcing_extension=True)
    fresh = _shim_state(child_cfg, monkeypatch)
    _fill_setup(fresh)
    fresh._nest_restart_classification = "REBUILT"
    restart._validate_restart(
        child_path, fresh, child_cfg, sealed_forcing_extension=True)
    fresh.thb.flat[0] += np.float32(1.0)
    with pytest.raises(restart.RestartMismatchError, match="child/nest"):
        restart._validate_restart(
            child_path, fresh, child_cfg, sealed_forcing_extension=True)

    unexpected_child_prefix = _rewrite_header(
        child_path, tmp_path / "child-prefix.npz",
        lambda header: header.__setitem__(
            "lateral_boundary_prefix",
            restart.lateral_boundary_prefix_identity(source)))
    with pytest.raises(restart.RestartMismatchError,
                       match="nested child unexpectedly"):
        restart._validate_restart(
            unexpected_child_prefix, child, child_cfg,
            sealed_forcing_extension=True)


def _sealed_tree_fixture(monkeypatch, *, forcing_count: int,
                         run_seconds: float, payload_seed: int):
    from gpuwm.core.model import ModelRuntimeStatus

    root_cfg = _cfg(
        grid_id=1, moist=True, mp_physics=1,
        run_seconds=run_seconds, restart_interval_s=3600.0,
        specified=True, spec_bdy_width=3, spec_zone=1, relax_zone=2)
    child_cfg = _cfg(
        grid_id=2, nested=True, moist=True, mp_physics=1,
        run_seconds=run_seconds, restart_interval_s=3600.0)
    root_state = _extension_state(
        root_cfg, monkeypatch, count=forcing_count)
    child_state = _shim_state(child_cfg, monkeypatch)
    _fill_setup(child_state)
    child_state._nest_restart_classification = "REBUILT"
    _fill_serialized(root_state, payload_seed)
    _fill_serialized(child_state, payload_seed + 1)

    start = datetime(2026, 7, 30)

    def clock(grid_id):
        return SimpleNamespace(
            ticks=3600, tick_den=1, step_count=3600,
            dtbc_fp32=np.float32(0.0),
            spec=SimpleNamespace(
                grid_id=grid_id, start_ticks=0, step_ticks=1),
            history_due=lambda: False)

    root_clock = clock(1)
    # CPU fixture analogue of attach + bind: validation only needs the
    # resident mirror's semantic identity; the host forcing inventory above
    # supplies the exact prefix bytes.
    root_state._lateral_boundary_device = SimpleNamespace(
        rolling=False, clock=root_clock)
    root = SimpleNamespace(
        cfg=SimpleNamespace(
            grid_id=1, parent_id=0, run=root_cfg, start_time=None),
        state=root_state, clock=root_clock, parent=None, children=[],
        coupler=None, _started=True)

    class Coupler:
        valid = True

        def invalidate(self):
            self.valid = False

    child = SimpleNamespace(
        cfg=SimpleNamespace(
            grid_id=2, parent_id=1, run=child_cfg, start_time=None),
        state=child_state, clock=clock(2), parent=root, children=[],
        coupler=Coupler(), _started=True)
    root.children.append(child)
    model = SimpleNamespace(
        root=root,
        experiment_fingerprint="sealed-tree-fixture-v1",
        schedule=SimpleNamespace(
            period_ticks=1,
            clock=SimpleNamespace(
                tick_den=1, run_ticks=int(run_seconds), start_time=start)),
        walk_parent_first=lambda: iter((root, child)),
        _runtime_status=ModelRuntimeStatus(),
        _io_manager=None,
        _last_checkpoint=None,
        _resumed=False,
        _resume_committed_history_grid_ids=frozenset())
    return model, start


def test_sealed_tree_restart_extends_root_and_restores_rebuilt_child(
        monkeypatch, tmp_path):
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0,
        payload_seed=31)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600),
        sealed_forcing_extension=True)
    assert restart.read_restart_header(root_path)[
        "root_external_lbc_clock"] == \
        restart.ROOT_EXTERNAL_LBC_CLOCK_IDENTITY

    resumed, _ = _sealed_tree_fixture(
        monkeypatch, forcing_count=3, run_seconds=7200.0,
        payload_seed=91)
    info = restart.restore_tree_restart(
        root_path, resumed, sealed_forcing_extension=True)
    assert info.elapsed_ticks == 3600
    assert np.array_equal(resumed.root.state.u, source.root.state.u,
                          equal_nan=True)
    resumed_child = tuple(resumed.walk_parent_first())[1]
    source_child = tuple(source.walk_parent_first())[1]
    assert np.array_equal(resumed_child.state.u, source_child.state.u,
                          equal_nan=True)
    assert resumed_child.coupler.valid is False


def _grell_freitas_tree(monkeypatch, *, seed=61):
    """Two ``cu_physics = 3`` domains on the CPU shim, legal for
    :func:`write_tree_restart`.  No device: the manifest walk that refused
    the real run is pure Python."""
    from gpuwm.core.gf import GrellFreitas
    from gpuwm.core.model import ModelRuntimeStatus

    def build(grid_id, nested, payload):
        cfg = _cfg(grid_id=grid_id, nested=nested, moist=True,
                   cu_physics=3, cudt_minutes=0.0, run_seconds=60.0)
        state, driver = _shim_driver_state(cfg, monkeypatch)
        _fill_setup(state)
        _fill_serialized(state, payload)
        state._nest_restart_classification = "REBUILT"
        adapter = GrellFreitas()
        # Exactly what PhysicsDriver._run_cumulus does on every cumulus
        # step, and therefore the state every real checkpoint sees.
        adapter.bind_driver(driver)
        driver.cumulus_callable = adapter
        return cfg, state

    def clock(dtbc):
        return SimpleNamespace(
            ticks=0, tick_den=1, step_count=0,
            dtbc_fp32=np.float32(dtbc),
            spec=SimpleNamespace(start_ticks=0, step_ticks=1))

    root_cfg, root_state = build(1, False, seed)
    child_cfg, child_state = build(2, True, seed + 1)
    root = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1, parent_id=0, run=root_cfg,
                            start_time=None),
        state=root_state, clock=clock(7.0), parent=None, children=[],
        coupler=None, _started=True)
    child = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=2, parent_id=1, run=child_cfg,
                            start_time=None),
        state=child_state, clock=clock(9.0), parent=root, children=[],
        coupler=None, _started=True)
    root.children.append(child)
    return SimpleNamespace(
        root=root, experiment_fingerprint="gf-tree-fixture-v1",
        schedule=SimpleNamespace(
            period_ticks=1,
            clock=SimpleNamespace(tick_den=1, run_ticks=60,
                                  start_time=datetime(1974, 4, 3, 12))),
        walk_parent_first=lambda: iter((root, child)),
        _runtime_status=ModelRuntimeStatus(), _io_manager=None,
        _last_checkpoint=None, _resumed=False,
        _resume_committed_history_grid_ids=frozenset())


def test_tree_checkpoint_publishes_with_grell_freitas_on_every_domain(
        monkeypatch, tmp_path):
    """The path the real run actually died on, reproduced without a device.

    The production failure was a TREE checkpoint of two GF domains --
    ``write_tree_restart`` -> ``write_restart`` -> ``_driver_manifest`` ->
    ``_callable_state_check`` -- and the tree writer publishes nothing
    unless every member writes, so one unclassified attribute on one
    domain refused the entire generation and unlinked what it had already
    written.  The single-domain bitwise gate cannot see that: it never
    calls the tree writer.

    Both adapters are bound the way ``_run_cumulus`` binds them, because
    an unbound adapter is exactly the state that WOULD have passed.
    """
    model = _grell_freitas_tree(monkeypatch)
    for node in model.walk_parent_first():
        assert node.state.physics.cumulus_callable._driver is \
            node.state.physics

    root_path = restart.write_tree_restart(
        tmp_path, model, datetime(1974, 4, 3, 15))

    published = sorted(path.name for path in tmp_path.glob("gpuwmrst_*.npz"))
    assert len(published) == 2, published
    assert root_path.exists()
    for path in tmp_path.glob("gpuwmrst_*.npz"):
        header = restart.read_restart_header(path)
        assert header["physics_setup"]["cumulus"]["callable"] == {
            "class": "gpuwm.core.gf.GrellFreitas",
            "implementation": "stock"}
        # GF ships no coefficient table; the KF asset must not be bound
        # into a GF manifest.
        assert header["physics_setup"]["cumulus"]["coefficient_table"] is None
        assert "cumulus/w0avg" not in header["array_manifest"]

    # And the negative, on the same tree: restore the pre-fix
    # classification and the whole generation is refused, publishing
    # nothing -- the exact production failure, message included.
    refused = _grell_freitas_tree(monkeypatch, seed=71)
    monkeypatch.setattr(restart, "CUMULUS_CALLABLE_CONTAINERS",
                        frozenset({"_history_state"}))
    empty = tmp_path / "refused"
    with pytest.raises(restart.RestartManifestError,
                       match=r"'_driver' is an object container"):
        restart.write_tree_restart(empty, refused, datetime(1974, 4, 3, 15))
    assert not list(empty.glob("gpuwmrst_*.npz"))


def test_sealed_tree_writer_preflights_all_members_before_publication(
        monkeypatch, tmp_path):
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0,
        payload_seed=51)
    for node in source.walk_parent_first():
        node.clock.ticks = 3500

    with pytest.raises(restart.RestartMismatchError, match="not sealed"):
        restart.write_tree_restart(
            tmp_path, source, start + timedelta(seconds=3600),
            sealed_forcing_extension=True)

    assert not list(tmp_path.glob("gpuwmrst_*.npz"))


def _rewrite_header(path, output, mutate) -> Path:
    """Rewrite the JSON header only, leaving every array member intact."""
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))
    mutate(header)
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    with output.open("wb") as stream:
        np.savez(stream, **payload)
    return output


def _external_mirror(*, bound: bool):
    """The binding-status surface the restart identity rule consults.

    ``attach_lateral_boundaries`` leaves ``clock=None`` (unbound legacy
    consumption); the production tree build assigns the root DomainClock
    (bound WRF post-increment consumption)."""
    return SimpleNamespace(rolling=False,
                           clock=object() if bound else None)


def _external_boundaries(seed: int):
    from gpuwm.ingest.lateral_bc import build_lateral_boundaries

    rng = np.random.default_rng(seed)
    snapshots = [{"u": rng.standard_normal((8, 8))} for _ in range(2)]
    return build_lateral_boundaries(
        snapshots, [0.0, 3600.0], spec_bdy_width=3, spec_zone=1,
        relax_zone=2)


def test_specified_restart_header_carries_root_lbc_clock_identity(
        monkeypatch, tmp_path):
    """Davies-bind restart epoch (dossier section 6): a specified domain's
    checkpoint records WHICH root external-LBC clock semantic integrated
    its trajectory.  Binding is invisible to the config echo, the setup
    fingerprint, and the physics identity, so without this key an old
    elapsed-based (pre-bind) checkpoint would silently resume under
    WRF-post-increment semantics -- a different trajectory."""
    cfg = _cfg(moist=True, mp_physics=1, specified=True)

    bound = _shim_state(cfg, monkeypatch)
    _fill_setup(bound)
    bound.lateral_boundaries = _external_boundaries(1)
    bound._lateral_boundary_device = _external_mirror(bound=True)
    bound_path = restart.write_restart(tmp_path / "bound.npz", bound, cfg)
    header = restart.read_restart_header(bound_path)
    assert header.get("root_external_lbc_clock") == \
        restart.ROOT_EXTERNAL_LBC_CLOCK_IDENTITY
    assert restart.ROOT_EXTERNAL_LBC_CLOCK_IDENTITY == "wrf-postincrement-v1"

    unbound = _shim_state(cfg, monkeypatch)
    _fill_setup(unbound)
    unbound.lateral_boundaries = _external_boundaries(1)
    unbound._lateral_boundary_device = _external_mirror(bound=False)
    legacy_path = restart.write_restart(
        tmp_path / "legacy.npz", unbound, cfg)
    legacy_header = restart.read_restart_header(legacy_path)
    assert legacy_header.get("root_external_lbc_clock") == \
        restart.ROOT_EXTERNAL_LBC_CLOCK_LEGACY
    assert restart.ROOT_EXTERNAL_LBC_CLOCK_LEGACY == "legacy-elapsed-v0"

    # Non-specified domains have no external Davies consumer: no key.
    plain_cfg = _cfg(moist=True, mp_physics=1)
    plain = _shim_state(plain_cfg, monkeypatch)
    _fill_setup(plain)
    plain_path = restart.write_restart(
        tmp_path / "plain.npz", plain, plain_cfg)
    assert "root_external_lbc_clock" not in restart.read_restart_header(
        plain_path)


def test_restore_refuses_root_lbc_clock_semantic_mismatch(
        monkeypatch, tmp_path):
    """Old unbound checkpoints fail closed under bound production code,
    and post-bind checkpoints fail closed on legacy unbound paths.  A
    header with NO identity key is a pre-epoch file and is treated as
    legacy-elapsed (the semantics every pre-bind writer integrated)."""
    cfg = _cfg(moist=True, mp_physics=1, specified=True)

    def specified_state(*, bound: bool):
        state = _shim_state(cfg, monkeypatch)
        _fill_setup(state)
        state.lateral_boundaries = _external_boundaries(1)
        state._lateral_boundary_device = _external_mirror(bound=bound)
        return state

    bound_path = restart.write_restart(
        tmp_path / "bound.npz", specified_state(bound=True), cfg)
    legacy_path = restart.write_restart(
        tmp_path / "legacy.npz", specified_state(bound=False), cfg)
    pre_epoch = _rewrite_header(
        bound_path, tmp_path / "pre-epoch.npz",
        lambda header: header.pop("root_external_lbc_clock"))

    # Matching semantics restore: bound file -> bound resuming state.
    # (The shim mirror satisfies the attach requirement; the resident
    # device tables themselves are rebuilt by preparation, not restored.)
    restart.restore_restart(bound_path, specified_state(bound=True), cfg)
    # Legacy file -> legacy resuming state also remains valid.
    restart.restore_restart(legacy_path, specified_state(bound=False), cfg)

    with pytest.raises(restart.RestartMismatchError,
                       match="root_external_lbc_clock"):
        restart.restore_restart(
            legacy_path, specified_state(bound=True), cfg)
    with pytest.raises(restart.RestartMismatchError,
                       match="root_external_lbc_clock"):
        restart.restore_restart(
            pre_epoch, specified_state(bound=True), cfg)
    with pytest.raises(restart.RestartMismatchError,
                       match="root_external_lbc_clock"):
        restart.restore_restart(
            bound_path, specified_state(bound=False), cfg)


def test_truncated_restart_file_is_rejected_loudly(monkeypatch, tmp_path):
    """Review F4: corrupt/truncated archives raise a gpuwm-branded error
    naming the file and the likely cause, and writes publish atomically
    (no .tmp residue, never a truncated file under the final name)."""
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    path = restart.write_restart(tmp_path / "rst.npz", state, cfg)
    assert not list(tmp_path.glob("*.tmp"))

    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes(path.read_bytes()[:256])
    with pytest.raises(restart.RestartMismatchError,
                       match="truncated or corrupt"):
        restart.read_restart_header(truncated)
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    with pytest.raises(restart.RestartMismatchError,
                       match="truncated or corrupt"):
        restart.restore_restart(truncated, fresh, cfg)


def test_tree_restore_corrupt_later_member_leaves_every_domain_untouched(
        monkeypatch, tmp_path):
    """Shadow #2: d01 may be valid while a d02 payload member has a bad
    CRC.  Its header remains readable, but complete-set validation must refuse
    before applying even one d01 byte or changing any clock/coupler metadata.
    """
    cfg1 = _cfg(grid_id=1, moist=False, run_seconds=60.0)
    cfg2 = _cfg(grid_id=2, nested=True, moist=False, run_seconds=60.0)

    def state(cfg, seed):
        value = _shim_state(cfg, monkeypatch)
        value._nest_restart_classification = "REBUILT"
        _fill_setup(value)
        _fill_serialized(value, seed)
        return value

    source1, source2 = state(cfg1, 1), state(cfg2, 2)
    common = {
        "experiment_fingerprint": "complete-set-fixture",
        "checkpoint_set_id": "one-set",
        "domain_ids": [1, 2],
        "elapsed_ticks": 0,
        "tick_den": 1,
        "phase": "PERIOD_BEGIN",
        "nest_tables": "REBUILT",
        "dtbc_fp32_bits": int(np.float32(0.0).view(np.uint32)),
    }
    path1 = restart.write_restart(
        tmp_path / "gpuwmrst_d01_1982-05-20_00_00_00.npz",
        source1, cfg1,
        tree_header={**common, "grid_id": 1, "parent_id": 0})
    path2 = restart.write_restart(
        tmp_path / "gpuwmrst_d02_1982-05-20_00_00_00.npz",
        source2, cfg2,
        tree_header={**common, "grid_id": 2, "parent_id": 1})

    # Flip one stored member byte without rewriting its ZIP CRC. np.load can
    # still read the independent JSON header member, but loading state/u fails.
    with zipfile.ZipFile(path2) as archive:
        member = archive.getinfo("state/u.npy")
    raw = bytearray(path2.read_bytes())
    local = raw[member.header_offset:member.header_offset + 30]
    fields = struct.unpack("<IHHHHHIIIHH", local)
    payload_start = member.header_offset + 30 + fields[-2] + fields[-1]
    raw[payload_start + member.file_size // 2] ^= 0x01
    path2.write_bytes(raw)
    assert restart.read_restart_header(path2)["grid_id"] == 2

    live1, live2 = state(cfg1, 11), state(cfg2, 12)

    class Coupler:
        valid = True

        def invalidate(self):
            self.valid = False

    root_cfg = SimpleNamespace(grid_id=1, parent_id=0, run=cfg1)
    child_cfg = SimpleNamespace(grid_id=2, parent_id=1, run=cfg2)
    clock1 = SimpleNamespace(
        ticks=0, tick_den=1, step_count=0, dtbc_fp32=np.float32(7.0),
        spec=SimpleNamespace(step_ticks=1))
    clock2 = SimpleNamespace(
        ticks=0, tick_den=1, step_count=0, dtbc_fp32=np.float32(9.0),
        spec=SimpleNamespace(step_ticks=1))
    root = SimpleNamespace(
        cfg=root_cfg, state=live1, clock=clock1, parent=None, coupler=None)
    child = SimpleNamespace(
        cfg=child_cfg, state=live2, clock=clock2, parent=root,
        coupler=Coupler())

    model = SimpleNamespace(
        experiment_fingerprint="complete-set-fixture", root=root,
        schedule=SimpleNamespace(
            period_ticks=1,
            clock=SimpleNamespace(tick_den=1, run_ticks=60)),
        walk_parent_first=lambda: iter((root, child)))

    before = {
        gid: {name: getattr(value, name).tobytes()
              for name in restart.STATE_SERIALIZED_ATTRS
              if getattr(value, name, None) is not None}
        for gid, value in ((1, live1), (2, live2))}
    metadata = (clock1.ticks, clock1.step_count, clock1.dtbc_fp32.tobytes(),
                clock2.ticks, clock2.step_count, clock2.dtbc_fp32.tobytes(),
                child.coupler.valid, live1.elapsed_seconds,
                live2.elapsed_seconds)
    with pytest.raises(restart.RestartMismatchError,
                       match="truncated or corrupt"):
        restart.restore_tree_restart(path1, model)

    after = {
        gid: {name: getattr(value, name).tobytes()
              for name in restart.STATE_SERIALIZED_ATTRS
              if getattr(value, name, None) is not None}
        for gid, value in ((1, live1), (2, live2))}
    assert after == before
    assert metadata == (
        clock1.ticks, clock1.step_count, clock1.dtbc_fp32.tobytes(),
        clock2.ticks, clock2.step_count, clock2.dtbc_fp32.tobytes(),
        child.coupler.valid, live1.elapsed_seconds, live2.elapsed_seconds)


def test_failed_write_leaves_no_residue(monkeypatch, tmp_path):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(restart.np, "savez", explode)
    with pytest.raises(OSError, match="disk full"):
        restart.write_restart(tmp_path / "rst.npz", state, cfg)
    assert list(tmp_path.iterdir()) == []


def test_read_header_rejects_non_restart_npz(tmp_path):
    path = tmp_path / "plain.npz"
    with path.open("wb") as stream:
        np.savez(stream, some_array=np.zeros(3))
    with pytest.raises(restart.RestartMismatchError, match="header"):
        restart.read_restart_header(path)


def test_restart_filename_is_wrfrst_style():
    assert (restart.restart_filename(datetime(1974, 4, 3, 18))
            == "gpuwmrst_d01_1974-04-03_18_00_00.npz")


# ---------------------------------------------------------------------------
# Config knob and CLI plumbing (CPU).
# ---------------------------------------------------------------------------

def test_restart_interval_validation():
    from gpuwm.config import load_config
    from gpuwm.verify.cases.real74_d01 import (_restart_outer_steps,
                                               phase3_config)

    cfg = phase3_config()
    assert _restart_outer_steps(cfg) is None                    # default off
    assert _restart_outer_steps(
        replace(cfg, restart_interval_s=21600.0)) == 360
    with pytest.raises(ValueError, match="restart_interval_s"):
        _restart_outer_steps(replace(cfg, restart_interval_s=90.5))

    import tempfile
    text = (REPO_ROOT / "configs" / "real74_d01.toml").read_text(
        encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.toml"
        bad.write_text(text + "\nrestart_interval_s = -5.0\n",
                       encoding="utf-8")
        with pytest.raises(ValueError, match="restart_interval_s"):
            load_config(bad)
        good = Path(tmp) / "good.toml"
        good.write_text(text + "\nrestart_interval_s = 21600.0\n",
                        encoding="utf-8")
        assert load_config(good).restart_interval_s == 21600.0


def test_cli_run_passes_restart_through(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    import gpuwm.cli as cli

    calls = []
    cfg = SimpleNamespace(case="real74_d01")
    fake = SimpleNamespace(
        run_config=lambda loaded, outdir, restart=None: calls.append(
            (loaded, outdir, restart)) or SimpleNamespace(
                wrfout_paths=(), completed_seconds=43200.0, nan_free=True))
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setitem(cli._REAL_CASES, "real74_d01", fake)
    restart_file = tmp_path / "gpuwmrst_d01_1974-04-03_18_00_00.npz"
    # The config path must EXIST even though `load_config` is mocked:
    # a4417efb refuses a path that is not a readable regular file before
    # anything opens it, so a name that was never written now exits 2
    # ("does not exist; pass the experiment .toml that `gpuwm domain`
    # wrote") without reaching the loader this test replaces.  A legacy
    # [run]-shaped body keeps the CLI on the registered-case branch the
    # rest of the test asserts on.
    config = tmp_path / "cfg.toml"
    config.write_text("[run]\ncase = \"real74_d01\"\n", encoding="utf-8")

    assert cli.main(["run", str(config),
                     "--outdir", str(tmp_path / "run"),
                     "--restart", str(restart_file)]) == 0
    assert calls == [(cfg, tmp_path / "run", restart_file)]
    assert "'restarted': True" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# GPU: live manifest coverage and the cheap bit-identity gate.
# ---------------------------------------------------------------------------

def _physics_state(cp, **cfg_overrides):
    """Small full-physics state: Morrison + analytic radiation + KF +
    SFCLAY/Noah/YSU, mixed land/water, with a condensate layer so the
    microphysics accumulators and h_diabatic are non-trivial.

    ``cp`` is the caller's cupy module, passed in rather than imported here.
    See :func:`_grell_freitas_state` for why that matters.
    """
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics

    values = dict(nx=6, ny=4, nz=16, dx=2000.0, dy=2000.0,
                  ztop=8000.0, dt=10.0, run_seconds=0.0,
                  time_step_sound=4, moist=True, mp_physics=10,
                  sf_sfclay_physics=1, sf_surface_physics=2,
                  bl_pbl_physics=1, ra_physics=90, cu_physics=1,
                  radt_minutes=1.0, cudt_minutes=1.0, bldt=0.0)
    values.update(cfg_overrides)
    cfg = RunConfig(**values)
    coord = make_vertical_coord(cfg.nz)
    def theta(z):
        return 298.0 + 0.004 * np.asarray(z, np.float64)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.011 * np.exp(-np.asarray(z, np.float64) / 2400.0))
    state.u[...] = cp.float32(6.0)
    state.v[...] = cp.float32(0.5)
    state.qc[2:6] = cp.float32(4.0e-4)       # cloud layer -> rain/heating

    landmask = np.ones((cfg.ny, cfg.nx), np.float64)
    landmask[:, -1] = 0.0
    tsk = np.full((cfg.ny, cfg.nx), 299.0)
    tsk[:, -1] = 296.0
    soil_t = np.stack([tsk - 0.5, tsk - 1.0, tsk - 1.5, tsk - 2.0])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.31)
    soil_m[:, landmask == 0.0] = 1.0
    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m, ivgtyp=np.where(landmask, 10, 17),
        isltyp=np.where(landmask, 6, 14), vegfra=55.0, tmn=286.0,
        swdown=450.0, glw=310.0, pblh=700.0,
        radiation_start_time=datetime(1974, 4, 3, 12),
        radiation_latitude=np.full((cfg.ny, cfg.nx), 39.0),
        radiation_longitude=np.full((cfg.ny, cfg.nx), -87.0))
    return state, cfg, driver


def _grell_freitas_state(cp):
    """Small state in which ``cu_physics=3`` actually convects.

    Deliberately NOT ``_physics_state`` with an override: that state's 2 km
    grid and 8 km lid are a configuration Grell-Freitas declines (the
    ``sig = (1-frh)^2`` scale taper, and no room above the LFC), and a
    restart gate for a scheme that never triggers would compare nothing but
    zeros on the axis it exists to test.  This is the production family --
    12 km, RTE+RRTMGP, Morrison, YSU, ``cudt_minutes = 0`` -- over a warm,
    conditionally unstable, deep-moist column that GF fires on within the
    first ten steps and is still firing at step 40.  The 14 km lid is the
    RRTMGP temperature-range floor, not a physics choice.

    ``cp`` is the caller's cupy module.  It is a parameter, not an ``import
    cupy`` in this body, and that is load-bearing: conftest marks a module
    ``gpu`` *in its entirety* when any non-test function imports cupy,
    because a helper's callers are not decidable from the AST.  These three
    state builders are called only from the five explicitly
    ``@pytest.mark.gpu`` tests below, but the coarse rule cannot know that,
    so importing cupy here swept the module's other 91 device-free tests --
    every restart-manifest classification gate, including the two covering
    the Grell-Freitas manifest fix -- out of ``-m "not gpu"`` and made them
    unrunnable without a card.  Keeping the import in the callers puts the
    device contact where it actually happens.  Do not "tidy" it back.
    """
    from gpuwm.config import validate_run_config
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics

    cfg = validate_run_config(RunConfig(
        nx=8, ny=6, nz=40, dx=12000.0, dy=12000.0, ztop=14000.0,
        dt=60.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=10, morr_rimed_ice=1,
        sf_sfclay_physics=1, sf_surface_physics=2, bl_pbl_physics=1,
        # radt 6 min on a 60 s step -> stepra 6, so radiation is due at
        # itimestep % 6 == 1 and the step-20 checkpoint is OFF its calendar.
        ra_physics=4, radt_minutes=6.0, bldt=0.0,
        cu_physics=3, cudt_minutes=0.0, ishallow=1))
    coord = make_vertical_coord(cfg.nz)

    def theta(z):
        # Well-mixed to 2 km, then conditionally unstable.
        z = np.asarray(z, np.float64)
        return 300.0 + 0.0025 * np.maximum(z - 2000.0, 0.0)

    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.019 * np.exp(-np.asarray(z, np.float64) / 5000.0))
    state.u[...] = cp.float32(4.0)
    state.v[...] = cp.float32(1.0)

    landmask = np.ones((cfg.ny, cfg.nx), np.float64)
    landmask[:, -1] = 0.0                      # one water column: XLAND varies
    tsk = np.full((cfg.ny, cfg.nx), 308.0)
    tsk[:, -1] = 303.0
    soil_t = np.stack([tsk - 0.5, tsk - 1.0, tsk - 1.5, tsk - 2.0])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.35)
    soil_m[:, landmask == 0.0] = 1.0
    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m, ivgtyp=np.where(landmask, 10, 17),
        isltyp=np.where(landmask, 6, 14), vegfra=55.0, tmn=292.0,
        swdown=850.0, glw=380.0, pblh=1200.0,
        radiation_start_time=datetime(1974, 4, 3, 18),
        radiation_latitude=np.full((cfg.ny, cfg.nx), 39.0),
        radiation_longitude=np.full((cfg.ny, cfg.nx), -87.0))
    return state, cfg, driver


def _thompson_restart_state(cp):
    """Small guarded-mp8 state with all classic hydrometeor groups active.

    ``cp`` is the caller's cupy module.  See :func:`_grell_freitas_state`.
    """
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.state import init_theta_perturbation

    cfg = RunConfig(
        nx=16, ny=12, nz=24, dx=1000.0, dy=1000.0, ztop=12000.0,
        dt=2.0, run_seconds=40.0, time_step_sound=4,
        moist=True, mp_physics=8)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z, dtype=float),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_theta_perturbation(
        cfg, coord, base,
        lambda x, z: 0.20 * np.exp(-(np.asarray(z) / 4500.0) ** 2)[
            :, None, None])

    oracle = (REPO_ROOT / "gpuwm" / "data" / "thompson" / "oracle" /
              "mixed-column.csv")
    with oracle.open(newline="", encoding="ascii") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["phase"] == "before"]

    def profile(name):
        return np.asarray([float(row[name]) for row in rows], np.float32)

    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        values = profile(name)[:, None, None]
        getattr(state, name)[...] = cp.asarray(
            np.broadcast_to(values, (cfg.nz, cfg.ny, cfg.nx)))
    state.nr[...] = cp.asarray(np.broadcast_to(
        profile("nr_per_kg")[:, None, None], state.nr.shape))
    state.ni[...] = cp.asarray(np.broadcast_to(
        profile("ni_per_kg")[:, None, None], state.ni.shape))
    state.u[...] = cp.float32(6.0)
    initialize_physics(state, cfg)
    return state, cfg


def _advance_thompson_with_refl(state, cfg, steps):
    """Advance and consume the output-only handoff every fifth step."""
    from gpuwm.core.dycore import step
    from gpuwm.core.refl import consume_refl_10cm

    for _ in range(steps):
        step_index = int(round(state.elapsed_seconds / cfg.dt)) + 1
        due = step_index % 5 == 0
        step(state, cfg, refl_10cm_due=due)
        if due:
            refl = consume_refl_10cm(state)
            assert bool(np.isfinite(refl.get()).all())


def _assert_restart_equal(path_a, path_b, *, compare_trackers=False):
    """Bit-exact equality of two restart files (arrays + relevant header)."""
    with np.load(path_a, allow_pickle=False) as data:
        arrays_a = {key: data[key] for key in data.files}
    with np.load(path_b, allow_pickle=False) as data:
        arrays_b = {key: data[key] for key in data.files}
    header_key = "__gpuwm_restart_header__"
    header_a = json.loads(bytes(bytearray(arrays_a.pop(header_key))))
    header_b = json.loads(bytes(bytearray(arrays_b.pop(header_key))))

    assert set(arrays_a) == set(arrays_b), (
        sorted(set(arrays_a) ^ set(arrays_b)))
    for key in sorted(arrays_a):
        assert arrays_a[key].dtype == arrays_b[key].dtype, key
        np.testing.assert_array_equal(arrays_a[key], arrays_b[key],
                                      err_msg=key)
        assert arrays_a[key].tobytes() == arrays_b[key].tobytes(), key

    assert header_a["elapsed_seconds"] == header_b["elapsed_seconds"]
    assert header_a["setup_fingerprint"] == header_b["setup_fingerprint"]
    assert header_a["physics_setup_fingerprint"] == \
        header_b["physics_setup_fingerprint"]
    assert header_a["physics_setup"] == header_b["physics_setup"]
    assert header_a["driver"] == header_b["driver"]
    assert header_a["array_manifest"] == header_b["array_manifest"]
    for key, value in header_a["config"].items():
        if key in restart.CONFIG_RUN_LENGTH_FIELDS:
            continue
        assert header_b["config"][key] == value, key
    if compare_trackers:
        assert header_a["run_trackers"] == header_b["run_trackers"]


@requires_gpu
@pytest.mark.gpu
def test_live_full_state_write_covers_every_mechanism(tmp_path):
    """write_restart's internal completeness walk passes on a live
    full-physics state, and the manifest carries every audited mechanism."""
    import cupy as cp

    from gpuwm.core.dycore import run_steps

    state, cfg, driver = _physics_state(cp)
    run_steps(state, cfg, 6)
    path = restart.write_restart(tmp_path / "rst.npz", state, cfg)
    header = restart.read_restart_header(path)
    keys = set(header["array_manifest"])
    for expected in (
            "state/u", "state/thp", "state/php", "state/mup",
            "state/h_diabatic", "state/effc", "state/ng",
            "scratch/mp_rainnc", "scratch/mp_sr",
            "scratch/cu_rainc", "scratch/cu_nca", "scratch/cu_pratec",
            "scratch/cu_raincv", "scratch/cu_rthcuten",
            "driver/rthratenlw", "driver/rthratensw",
            "driver/pending_rainbl",
            "driver/pbl_tendencies/ru", "driver/radiation_tendencies/rtheta",
            "driver/cumulus_tendencies/rqv",
            "fields/tsk", "fields/ust", "fields/mol", "fields/znt",
            "fields/qsfc", "fields/hfx", "fields/qfx", "fields/pblh",
            "fields/sh2o", "fields/snotime", "fields/albedo",
            "fields/emiss", "fields/rainbl",
            "cumulus/w0avg"):
        assert expected in keys, expected
    assert not any(key.startswith("driver/microphysics/") for key in keys)
    assert header["driver"]["microphysics_updates"] == 6
    assert header["driver"]["call_counts"]["radiation"] >= 1


@requires_gpu
@pytest.mark.gpu
def test_short_full_physics_restart_is_bit_identical(tmp_path):
    """Cheap suite-resident gate: 20 steps + restart + 20 steps == 40
    steps, FP32-bit-exact on every serialized field.

    The 20-step boundary is deliberately OFF both physics calendars
    (radiation due at itimestep % 6 == 1 -> 19/25; cumulus due at
    itimestep % 6 == 0 -> 18/24), so the held radiation/cumulus
    tendencies, the KF NCA hold, W0AVG, and _pending_rainbl all span the
    restart boundary and must survive serialization, not recomputation.
    """
    import cupy as cp

    from gpuwm.core.dycore import run_steps

    state_a, cfg_a, _ = _physics_state(cp)
    run_steps(state_a, cfg_a, 40)
    reference = restart.write_restart(tmp_path / "reference.npz",
                                      state_a, cfg_a)

    state_b, cfg_b, _ = _physics_state(cp)
    run_steps(state_b, cfg_b, 20)
    mid = restart.write_restart(tmp_path / "mid.npz", state_b, cfg_b)

    state_c, cfg_c, _ = _physics_state(cp)
    info = restart.restore_restart(mid, state_c, cfg_c)
    assert info.elapsed_seconds == 200.0
    run_steps(state_c, cfg_c, 20)
    resumed = restart.write_restart(tmp_path / "resumed.npz",
                                    state_c, cfg_c)

    _assert_restart_equal(resumed, reference)
    # The comparison must be non-trivial: microphysics ran on every step
    # and produced retained heating; the surface cycle ran.
    assert state_c.physics.microphysics_updates == 40
    assert bool(cp.any(state_c.h_diabatic != 0.0))
    assert state_c.physics.call_counts["radiation"] == 7   # 1, 7, ..., 37
    assert state_c.physics.call_counts["cumulus"] == 7     # 1, 6, ..., 36


@requires_gpu
@pytest.mark.gpu
def test_short_grell_freitas_restart_is_bit_identical(tmp_path):
    """cu_physics=3, on the same terms: 20 + restart + 20 == 40 steps,
    FP32-bit-exact on every serialized field.

    The gate the GF selectable wiring shipped without, and the one that
    would have caught the manifest refusal that killed the first real-case
    GF trajectory 179 outer steps in.  GF's adapter is stateless, so
    everything that must cross the boundary crosses it on the DRIVER: the
    held ``cu_rates``, the RAINC/RAINCV/PRATEC accumulators, and --
    GF-specific -- the retained radiative heating that ``bind_driver``
    feeds back into GFDRV's forced state.  ``cudt_minutes=0`` puts GF on
    every step (its pinned WRF configuration), so the step-21 call is a
    real cumulus call reading restored inputs; and the step-20 boundary is
    off the radiation calendar (due at ``itimestep % 6 == 1`` -> 19/25), so
    that call must consume a DESERIALIZED rthraten rather than one the
    resume happened to recompute.
    """
    import cupy as cp

    from gpuwm.core.dycore import run_steps
    from gpuwm.core.gf import GrellFreitas

    state_a, cfg_a, driver_a = _grell_freitas_state(cp)
    assert isinstance(driver_a.cumulus_callable, GrellFreitas)
    run_steps(state_a, cfg_a, 40)
    # The adapter is bound by the time any checkpoint is taken -- which is
    # exactly the shape the manifest walk has to accept.
    assert driver_a.cumulus_callable._driver is driver_a
    reference = restart.write_restart(tmp_path / "reference.npz",
                                      state_a, cfg_a)

    state_b, cfg_b, driver_b = _grell_freitas_state(cp)
    run_steps(state_b, cfg_b, 20)
    # THE state the fix is about: at the checkpoint GF is convecting, so
    # held rates and a RAINC total both have to survive serialization.
    assert bool(cp.any(driver_b.cu_rates["rthcuten"] != 0.0))
    assert float(cp.abs(driver_b.rainc).max()) > 0.0
    mid = restart.write_restart(tmp_path / "mid.npz", state_b, cfg_b)

    state_c, cfg_c, _ = _grell_freitas_state(cp)
    info = restart.restore_restart(mid, state_c, cfg_c)
    assert info.elapsed_seconds == 1200.0
    run_steps(state_c, cfg_c, 20)
    resumed = restart.write_restart(tmp_path / "resumed.npz",
                                    state_c, cfg_c)

    _assert_restart_equal(resumed, reference)

    # Non-triviality: GF ran on every one of the 40 steps and left real
    # numbers behind, and the radiation it reads spans the boundary.
    assert state_c.physics.call_counts["cumulus"] == 40
    assert state_c.physics.call_counts["radiation"] == 7   # 1, 7, ..., 37
    assert float(cp.abs(state_c.physics.rainc).max()) > 0.0
    assert bool(cp.any(state_c.h_diabatic != 0.0))
    # The exact sum bind_driver hands GFDRV as its radiative forcing: if it
    # were zero the boundary-crossing this test is about would be vacuous.
    held_rthraten = state_c.physics.rthratenlw + state_c.physics.rthratensw
    assert bool(cp.any(held_rthraten != 0.0))

    keys = set(restart.read_restart_header(reference)["array_manifest"])
    for expected in ("scratch/cu_rainc", "scratch/cu_raincv",
                     "scratch/cu_pratec", "scratch/cu_rthcuten",
                     "scratch/cu_rqvcuten"):
        assert expected in keys, expected
    # GF carries no trigger history: the KF-only W0AVG member must not
    # appear, and its absence must not be mistaken for a lost field.
    assert "cumulus/w0avg" not in keys


@requires_gpu
@pytest.mark.gpu
def test_short_thompson_refl_restart_is_bit_identical(tmp_path):
    """Guarded mp8: 10 + restart + 10 equals uninterrupted 20 exactly.

    REFL_10CM is due/consumed four times.  Its private graupel-number shadow
    is rebuild-only, while all prognostics, held heating, and precipitation
    accumulators must cross the step-10 checkpoint without one changed bit.
    """
    if not os.environ.get("GPUWM_THOMPSON_TABLE_ROOT"):
        pytest.skip("GPUWM_THOMPSON_TABLE_ROOT is required for mp8 GPU gate")
    if os.environ.get("GPUWM_EXPERIMENTAL_THOMPSON_MP8") != "1":
        pytest.skip("GPUWM_EXPERIMENTAL_THOMPSON_MP8=1 is required")

    import cupy as cp

    control, control_cfg = _thompson_restart_state(cp)
    _advance_thompson_with_refl(control, control_cfg, 20)
    reference = restart.write_restart(
        tmp_path / "thompson-reference.npz", control, control_cfg)

    split, split_cfg = _thompson_restart_state(cp)
    _advance_thompson_with_refl(split, split_cfg, 10)
    mid = restart.write_restart(
        tmp_path / "thompson-mid.npz", split, split_cfg)

    resumed, resumed_cfg = _thompson_restart_state(cp)
    info = restart.restore_restart(mid, resumed, resumed_cfg)
    assert info.elapsed_seconds == 20.0
    _advance_thompson_with_refl(resumed, resumed_cfg, 10)
    resumed_path = restart.write_restart(
        tmp_path / "thompson-resumed.npz", resumed, resumed_cfg)

    _assert_restart_equal(resumed_path, reference)
    assert resumed.physics.refl_10cm is None
    assert resumed.physics.microphysics_updates == 20


# ---------------------------------------------------------------------------
# GPU + slow_acceptance: THE Task 8 gate (fresh-process real74 6h+6h == 12h).
# ---------------------------------------------------------------------------

def _write_case_config(tmp_path, name, run_seconds, restart_interval):
    text = (REPO_ROOT / "configs" / "real74_d01.toml").read_text(
        encoding="utf-8")
    text = text.replace("run_seconds = 43200.0",
                        f"run_seconds = {run_seconds}")
    text += f"restart_interval_s = {restart_interval}\n"
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _cli_run(config_path, outdir, restart_file=None):
    command = [sys.executable, "-m", "gpuwm.cli", "run", str(config_path),
               "--outdir", str(outdir)]
    if restart_file is not None:
        command += ["--restart", str(restart_file)]
    result = subprocess.run(command, capture_output=True, text=True,
                            cwd=REPO_ROOT)
    assert result.returncode == 0, (
        f"{' '.join(command)}\n--- stdout ---\n{result.stdout}"
        f"\n--- stderr ---\n{result.stderr}")
    return result


def _assert_wrfouts_equal(path_a, path_b):
    import netCDF4

    with netCDF4.Dataset(path_a) as a, netCDF4.Dataset(path_b) as b:
        assert set(a.variables) == set(b.variables), path_a
        for name in sorted(a.variables):
            va = np.asarray(np.ma.filled(a.variables[name][:], np.nan))
            vb = np.asarray(np.ma.filled(b.variables[name][:], np.nan))
            np.testing.assert_array_equal(va, vb,
                                          err_msg=f"{path_a.name}:{name}")


@requires_bundle
@requires_gpu
@pytest.mark.gpu
@pytest.mark.slow_acceptance
def test_real74_restart_continuation_is_bit_identical(tmp_path):
    """Plan Task 8 gate: real74 6 h -> write restart -> FRESH PROCESS ->
    6 h more == uninterrupted 12 h, FP32-bit-exact on every state field
    and accumulator, with identical run-summary trackers.

    Three CLI subprocesses (each a fresh Python/CUDA process):
      A: run_seconds=21600, restart_interval_s=21600 -> gpuwmrst @ 18Z.
      C: run_seconds=43200, restart_interval_s=21600 -> gpuwmrst @ 18Z
         and @ 00Z (the uninterrupted control).
      B: run_seconds=43200, restart_interval_s=43200, --restart A@18Z ->
         gpuwmrst @ 00Z (the resumed run).
    The restart files ARE the complete state serialization, so comparing
    them bit-for-bit compares every prognostic, every physics/soil/snow
    field, every accumulator, every held tendency, W0AVG/NCA, and the
    clock; the write-time A@18Z == C@18Z check additionally pins that
    writing a restart never perturbs the running state.
    """
    rst_6h = restart.restart_filename(datetime(1974, 4, 3, 18))
    rst_12h = restart.restart_filename(datetime(1974, 4, 4, 0))

    cfg_a = _write_case_config(tmp_path, "real74-a.toml", 21600.0, 21600.0)
    cfg_b = _write_case_config(tmp_path, "real74-b.toml", 43200.0, 43200.0)
    cfg_c = _write_case_config(tmp_path, "real74-c.toml", 43200.0, 21600.0)
    out_a = tmp_path / "run-a"
    out_b = tmp_path / "run-b"
    out_c = tmp_path / "run-c"

    _cli_run(cfg_a, out_a)
    _cli_run(cfg_c, out_c)
    result_b = _cli_run(cfg_b, out_b, restart_file=out_a / rst_6h)
    assert "'restarted': True" in result_b.stdout

    # Write-time identity: the 6 h state is independent of run_seconds and
    # of having written a restart mid-run.
    _assert_restart_equal(out_a / rst_6h, out_c / rst_6h,
                          compare_trackers=True)
    # THE continuation gate: fresh-process 6h+6h == uninterrupted 12h,
    # bit-exact on every serialized array, with identical gate trackers.
    _assert_restart_equal(out_b / rst_12h, out_c / rst_12h,
                          compare_trackers=True)

    # The resumed run's hourly products match the uninterrupted run's.
    for hour in range(19, 24):
        name = f"wrfout_d01_1974-04-03_{hour:02d}_00_00"
        _assert_wrfouts_equal(out_b / name, out_c / name)
    name = "wrfout_d01_1974-04-04_00_00_00"
    _assert_wrfouts_equal(out_b / name, out_c / name)
    # The resumed run does not rewrite the 12Z cold-start frame.
    assert not (out_b / "wrfout_d01_1974-04-03_12_00_00").exists()
    assert (out_c / "wrfout_d01_1974-04-03_12_00_00").exists()
