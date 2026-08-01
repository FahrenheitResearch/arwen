"""Memory-preflight estimator, scratch registry, nest manifest, N0 gates.

Phase-5 Task 11 (architecture section E).  Everything here except the
``gpu``-marked N0 allocation runs is pure CPU shape-formula arithmetic.

The golden byte pins are DELIBERATE: any change to the DomainState /
PhysicsDriver / scratch allocation surface must update the preflight
manifests AND these pins in the same diff, keeping the estimator an
enforced upper bound instead of a stale note.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import math
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gpuwm.config import RunConfig, load_config
from gpuwm.core import preflight as pf
from gpuwm.case_data import load_experiment_case
from gpuwm.experiment import build_experiment, experiment_from_run_config
from gpuwm.io import restart

ROOT = Path(__file__).resolve().parents[1]
CONFIG_4DOM = ROOT / "configs" / "real74_4dom.toml"
CONFIG_D01 = ROOT / "configs" / "real74_d01.toml"

GIB = pf.GIB

_TINY = dict(nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
             dt=1.0, run_seconds=10.0)


@pytest.fixture(scope="module")
def d01_cfg() -> RunConfig:
    return load_config(CONFIG_D01)


@pytest.fixture(scope="module")
def exp1(d01_cfg):
    return experiment_from_run_config(d01_cfg, datetime(1974, 4, 3, 12))


@pytest.fixture(scope="module")
def exp4():
    return load_experiment_case(CONFIG_4DOM)[0]


@pytest.fixture(scope="module")
def est4(exp4):
    return pf.estimate_experiment(exp4)


# ---------------------------------------------------------------------------
# (a) DomainState shape formulas
# ---------------------------------------------------------------------------

def test_state_manifest_matches_restart_classification(d01_cfg):
    """The state shape manifest and the restart manifest classify the SAME
    attribute set: a DomainState field added to state.py without updating
    both fails here (full-featured config exercises every conditional)."""
    nssl_cfg = dataclasses.replace(d01_cfg, mp_physics=18)
    names = (set(pf.state_array_shapes(d01_cfg))
             | set(pf.state_array_shapes(nssl_cfg)))
    classified = (set(restart.STATE_SERIALIZED_ATTRS)
                  | set(restart.STATE_REBUILT_ATTRS)
                  | set(restart.STATE_SETUP_ARRAYS))
    assert names == classified


def test_state_shape_formulas_staggering():
    cfg = RunConfig(**_TINY)
    shapes = pf.state_array_shapes(cfg)
    assert shapes["u"] == (4, 6, 9)
    assert shapes["v"] == (4, 7, 8)
    assert shapes["w"] == (5, 6, 8)
    assert shapes["php"] == (5, 6, 8)
    assert shapes["mup"] == (6, 8)
    assert shapes["msfu"] == (6, 9)
    # Flat terrain keeps 1-D base profiles (state.py:148-152).
    assert shapes["thb"] == (4,)
    assert shapes["phb"] == (5,)
    assert "qv" not in shapes and "h_diabatic" not in shapes
    terrain = pf.state_array_shapes(
        RunConfig(**_TINY, terrain_opt=1, moist=True, mp_physics=10))
    assert terrain["phb"] == (5, 6, 8)
    assert terrain["qv"] == (4, 6, 8)
    for name in ("qi", "ng0", "effs", "nc", "h_diabatic"):
        assert terrain[name] == (4, 6, 8)
    # Kessler moist: no Morrison moments, no nc.
    kessler = pf.state_array_shapes(
        RunConfig(**_TINY, moist=True, mp_physics=1))
    assert "qv" in kessler and "qi" not in kessler and "nc" not in kessler


def test_shared_dycore_state_symbols_are_restart_rebuilt_source(exp4):
    """The sharing registry is exactly the restart REBUILT authority."""
    assert pf.shared_dycore_state_symbols() == restart.STATE_REBUILT_ATTRS
    active = set().union(*(
        pf.state_array_shapes(dc.run).keys() for dc in exp4.domains))
    assert set(pf.shared_dycore_state_workspace_shapes(exp4.domains)) == (
        set(restart.STATE_REBUILT_ATTRS) & active)


def test_shared_dycore_state_workspace_binds_contiguous_prefixes(monkeypatch):
    """Every rebuilt field keeps its allocation shape/dtype/strides.

    Two differently sized Morrison domains bind each symbol to a C-contiguous
    prefix of the same per-symbol maximum backing.  The ordinary constructor
    remains independent for single-domain callers.
    """
    import types

    import gpuwm.core.state as state_mod

    monkeypatch.setattr(state_mod, "cp", np)
    cfg_small = RunConfig(
        **_TINY, moist=True, mp_physics=10)
    cfg_large = RunConfig(
        **{**_TINY, "nx": 11, "ny": 7, "nz": 5},
        moist=True, mp_physics=10)
    domains = (types.SimpleNamespace(run=cfg_small),
               types.SimpleNamespace(run=cfg_large))
    workspace = state_mod.build_shared_dycore_state_workspace(domains)

    active = (set(pf.state_array_shapes(cfg_small))
              | set(pf.state_array_shapes(cfg_large)))
    assert workspace.symbols == restart.STATE_REBUILT_ATTRS & active
    assert workspace.nbytes == pf.shared_dycore_state_workspace_bytes(domains)
    small = state_mod.DomainState(
        cfg_small, dycore_state_workspace=workspace)
    large = state_mod.DomainState(
        cfg_large, dycore_state_workspace=workspace)
    small_shapes = pf.state_array_shapes(cfg_small)
    large_shapes = pf.state_array_shapes(cfg_large)

    for name in sorted(workspace.symbols):
        small_value = getattr(small, name)
        large_value = getattr(large, name)
        assert small_value.shape == small_shapes[name]
        assert large_value.shape == large_shapes[name]
        assert small_value.dtype == large_value.dtype == np.dtype(np.float32)
        assert small_value.flags.c_contiguous
        assert large_value.flags.c_contiguous
        assert small_value.strides == np.empty(
            small_shapes[name], dtype=np.float32).strides
        assert large_value.strides == np.empty(
            large_shapes[name], dtype=np.float32).strides
        assert np.shares_memory(small_value, large_value)
        assert np.shares_memory(small_value, workspace.backing(name))

    default_a = state_mod.DomainState(cfg_small)
    default_b = state_mod.DomainState(cfg_small)
    assert not np.shares_memory(default_a.u0, default_b.u0)


def test_shared_dycore_state_workspace_rejects_concurrent_owners(
        monkeypatch):
    """A second domain turn cannot acquire the shared arrays concurrently."""
    import types

    import gpuwm.core.state as state_mod

    monkeypatch.setattr(state_mod, "cp", np)
    cfg = RunConfig(**_TINY, moist=True, mp_physics=10)
    workspace = state_mod.build_shared_dycore_state_workspace(
        (types.SimpleNamespace(run=cfg),))

    with workspace.acquire(("STEP", 1)):
        assert workspace.owner == ("STEP", 1)
        with pytest.raises(RuntimeError, match="owned.*STEP.*1"):
            with workspace.acquire(("FORCE", 2, 1)):
                pass
    assert workspace.owner is None


# ---------------------------------------------------------------------------
# (b) PhysicsDriver persistents
# ---------------------------------------------------------------------------

def test_physics_shapes_scheme_selection(d01_cfg, exp4):
    full = pf.physics_array_shapes(d01_cfg)
    nzs = (d01_cfg.nz, d01_cfg.ny, d01_cfg.nx)
    s2 = (d01_cfg.ny, d01_cfg.nx)
    # KF persistence + rqr growth are d01-only (cu_physics=1).
    assert full["cumulus/w0avg"] == nzs
    assert full["cumulus_tendencies/rqr"] == nzs
    assert full["pbl_tendencies/rqr"] == nzs
    assert not any(name.startswith("tendencies/") for name in full)
    # Morrison + YSU carries rqi through the bldt=0 in-place composition.
    assert full["pbl_tendencies/rqi"] == nzs
    assert full["radiation/latitude_deg"] == s2
    assert not any(name.startswith("microphysics/") for name in full)
    assert full["fields/smois"] == (4, d01_cfg.ny, d01_cfg.nx)
    # At bldt=0 the raw YSU dict is transient, not driver-persistent.
    assert not any(name.startswith("last_ysu/") for name in full)
    transient = pf.ysu_output_transient_shapes(d01_cfg)
    for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi",
                 "exch_h", "exch_m"):
        assert transient[f"ysu_output/{name}"] == nzs
    for name in ("hpbl", "kpbl", "wstar", "delta", "topdown_radsum",
                 "wstar3_2", "cloudflg"):
        assert transient[f"ysu_output/{name}"] == s2

    # Every positive cadence keeps the exact historical storage path.
    held = pf.physics_array_shapes(dataclasses.replace(d01_cfg, bldt=5.0))
    assert held["last_ysu/du"] == nzs
    assert held["last_ysu/cloudflg"] == s2
    assert held["tendencies/rqr"] == nzs
    assert held["tendencies/rqi"] == nzs
    # Once-per-process KF device LUT, counted on the cumulus domain.
    assert sum(4 * math.prod(shape) for name, shape in full.items()
               if "kf_lut" in name) == 441680
    # RRTMGP ozone climatology profiles (rrtmgp.py:1076-1077).
    assert sum(4 * math.prod(shape) for name, shape in full.items()
               if "_ozone" in name) == 480

    child = pf.physics_array_shapes(exp4.domain(2).run)
    assert "cumulus/w0avg" not in child
    assert "cumulus_tendencies/rqr" not in child
    assert not any("kf_lut" in name for name in child)
    assert "pbl_tendencies/rqi" in child
    assert not any(name.startswith(("last_ysu/", "tendencies/"))
                   for name in child)

    assert pf.physics_array_shapes(RunConfig(**_TINY)) == {}


def test_physics_manifest_groups_cover_restart_driver_attrs(d01_cfg):
    """Cross-pin against the restart driver manifest (Fable F4/F2): every
    array-bearing serialized driver attribute has a manifest group, and
    conditional/rebuilt driver arrays are cross-pinned to their lifetime
    decisions.  Active Morrison diagnostics are absent here because the
    driver aliases the separately counted serialized ``mp_*`` scratch set."""
    groups = {name.split("/")[0] for name in
              pf.physics_array_shapes(d01_cfg)}
    scalar_attrs = {"microphysics_updates", "call_counts",
                    "ysu_nan_guard_fires"}
    assert set(restart.DRIVER_SERIALIZED_ATTRS) - scalar_attrs <= groups
    assert "last_ysu" not in groups
    assert "last_ysu" in restart.DRIVER_REBUILT_ATTRS
    assert "microphysics" not in restart.DRIVER_SERIALIZED_ATTRS
    assert "microphysics" in restart.DRIVER_REBUILT_ATTRS


def test_physics_lifetime_audit_is_exact_name_closed_world(d01_cfg):
    names = [name for row in pf.PHYSICS_ARRAY_LIFETIME_AUDIT
             for name in row.names]
    assert len(names) == len(set(names)) == 56
    assert {row.disposition for row in pf.PHYSICS_ARRAY_LIFETIME_AUDIT} == {
        "transient_when_bldt_zero", "aliases_serialized_scratch",
        "aliases_fresh_pbl_at_bldt_zero", "retained_family_state"}
    for prefix, components in {
            "last_ysu": {"du", "dv", "dtheta", "dqv", "dqc", "dqi",
                         "exch_h", "exch_m", "hpbl", "kpbl", "wstar",
                         "delta", "topdown_radsum", "wstar3_2", "cloudflg"},
            "microphysics": set(restart.MICROPHYSICS_COMPONENTS),
            "tendencies": set(restart.TENDENCY_COMPONENTS)}.items():
        assert {name.split("/", 1)[1] for name in names
                if name.startswith(prefix + "/")} == components
    for name in names:
        assert pf.physics_array_lifetime(name) is not None
    assert pf.physics_array_lifetime("last_ysu/future_component") is None


def test_physics_fields_union_covers_sources():
    from gpuwm.core.noah import _F2D
    from gpuwm.core.sfclay import SFCLAY_OUTPUTS

    names = set(pf.physics_field_names_2d())
    assert set(SFCLAY_OUTPUTS) <= names
    assert set(_F2D) <= names
    assert {"ebal", "kpbl", "landmask", "xland", "lakemask"} <= names


# ---------------------------------------------------------------------------
# (c) Scratch-slot registry + completeness over call sites
# ---------------------------------------------------------------------------

def test_scratch_registry_feature_matrix(d01_cfg):
    d01 = pf.scratch_slot_registry(d01_cfg, n_lbc_intervals=2)
    m = (d01_cfg.nz, d01_cfg.ny, d01_cfg.nx)
    assert d01["cu_rthcuten"] == m
    assert d01["morr_z8w"] == (d01_cfg.nz + 1, d01_cfg.ny, d01_cfg.nx)
    assert d01["pd_fxl"] == (d01_cfg.nz, d01_cfg.ny, d01_cfg.nx + 1)
    assert d01["lbc_qv_held"] == m
    assert d01["smag_rqi"] == m and d01["smag_rng"] == m
    assert d01["diff6_m"] == m
    assert d01["acoustic_mudf"] == (d01_cfg.ny, d01_cfg.nx)  # emdiv=0.01
    assert d01["integration_health_partial"] == (256, 9)
    assert d01["integration_health_field_ptr"] == (2048,)
    assert d01["integration_health_aux_ptr"] == (2048,)
    assert d01["integration_health_field_size"] == (2048,)
    assert d01["integration_health_bounds"] == (1024, 2)
    assert d01["integration_health_flags"] == (1024,)
    assert d01["integration_health_planes"] == (1024,)
    assert d01["integration_health_status_bits"] == (2048,)
    assert d01["integration_health_validation"] == (4,)
    assert d01["physics_validation_status"] == (1,)
    assert d01["lbc_old_mup_frame_1"] == (
        pf._perimeter_count(d01_cfg.ny, d01_cfg.nx, 1),)
    assert d01["lbc_forcing_tables"] == (2 * pf.lbc_interval_values(d01_cfg),)
    assert "mp_th" not in d01 and "openbc_upp_faces" not in d01

    kessler = pf.scratch_slot_registry(RunConfig(
        **_TINY, moist=True, mp_physics=1, open_x=True, open_y=True,
        khdif=1.0, kvdif=1.0))
    assert kessler["mp_kessler_sr"] == (6, 8)
    assert kessler["openbc_upp_faces"] == (4, 6, 2)
    assert kessler["openbc_vpp_faces"] == (4, 2, 8)
    assert kessler["diff_u"] == (4, 6, 9)
    assert kessler["physics_validation_status"] == (1,)
    # Open boundaries: PD final stage disabled -> no pd_* slots; not
    # specified -> no LBC residents.
    assert "pd_fxl" not in kessler and "lbc_relax_u" not in kessler
    assert "cu_nca" not in kessler

    dry_phys = pf.scratch_slot_registry(
        RunConfig(**_TINY, sf_sfclay_physics=1))
    assert dry_phys["physics_dry_qv"] == (4, 6, 8)
    assert dry_phys["physics_qi"] == (4, 6, 8)
    assert dry_phys["physics_qs"] == (4, 6, 8)
    assert "physics_validation_status" not in dry_phys

    kf_only = pf.scratch_slot_registry(
        RunConfig(**_TINY, moist=True, cu_physics=1))
    assert kf_only["physics_validation_status"] == (1,)
    assert pf._CUMULUS_KERNEL_MODULES[1] == ("kf", "kf_validation")


def test_nwp_diagnostics_prices_exactly_the_uh_planes(d01_cfg):
    """The UP_HELI_MAX lane costs three (ny, nx) FP32 planes and nothing
    else; the flagship (nwp_diagnostics = 0) registry is untouched."""
    base = pf.scratch_slot_registry(d01_cfg, n_lbc_intervals=2)
    on_cfg = dataclasses.replace(d01_cfg, nwp_diagnostics=1)
    on = pf.scratch_slot_registry(on_cfg, n_lbc_intervals=2)
    added = {"up_heli_max", "uh_diag_col", "uh_diag_use"}
    assert set(on) - set(base) == added
    assert not added & set(base)
    for slot in added:
        assert on[slot] == (d01_cfg.ny, d01_cfg.nx)


def test_scratch_registry_classifiable_by_restart_manifest(d01_cfg):
    """Every registry slot must already be classified by the restart
    manifest (serialize or rebuild) -- one namespace, two manifests, no
    drift.  ``nest_*`` slots are excluded: their REBUILT classification
    is Task 14's restart.py edit (architecture 'WRF deviations')."""
    union: set[str] = set()
    for cfg in (d01_cfg,
                RunConfig(**_TINY, moist=True, mp_physics=1, open_x=True,
                          open_y=True, khdif=1.0, kvdif=1.0, emdiv=0.01),
                RunConfig(**_TINY, sf_sfclay_physics=1)):
        union |= set(pf.scratch_slot_registry(cfg, n_lbc_intervals=2))
    for slot in union:
        assert restart.classify_scratch_slot(slot) in ("serialize", "rebuild")


def test_scratch_lifetime_audit_covers_registry_and_manifest(d01_cfg, exp4):
    """Architecture-E lever-2 admission is closed-world and reviewed.

    Every possible registry slot exercised by the feature matrix, plus every
    frozen F4 ``nest_*`` manifest slot, maps to exactly one committed audit
    row. Only explicit write-before-read rows may enter the arena.
    """
    configs = [dc.run for dc in exp4.domains]
    configs += [
        d01_cfg,
        RunConfig(**_TINY, moist=True, mp_physics=1, open_x=True,
                  open_y=True, khdif=1.0, kvdif=1.0, emdiv=0.01),
        RunConfig(**_TINY, sf_sfclay_physics=1),
    ]
    slots = set()
    for cfg in configs:
        slots |= set(pf.scratch_slot_registry(cfg, n_lbc_intervals=2))
    for manifest in pf.nest_allocation_manifest(exp4).values():
        slots |= set(manifest)

    assert slots
    for slot in slots:
        row = pf.scratch_slot_lifetime(slot)
        assert row is not None, slot
        assert row.kind in {"write_before_read", "carrying",
                            "excluded_unproven"}
        assert bool(pf.scratch_slot_uses_arena(slot)) == (
            row.kind == "write_before_read")
        if row.arena_eligible:
            assert row.evidence and row.rationale

    # High-risk exclusions are pins, not prefix accidents.
    for slot in ("mp_rainnc", "cu_rthcuten", "refl_10cm",
                 "physics_dry_qv", "lbc_forcing_tables", "nest_u_bxs"):
        assert not pf.scratch_slot_uses_arena(slot)
    for slot in ("rk_ru", "acoustic_c2a", "smag_rqi", "pd_fxl",
                 "morr_theta", "lbc_relax_u"):
        assert pf.scratch_slot_uses_arena(slot)


def test_shared_scratch_arena_aliases_views_and_default_does_not(monkeypatch):
    """Two different domain shapes share an admitted slot's max backing.

    The no-arena constructor retains the original independent, zero-filled
    per-state allocation path.
    """
    import types

    import gpuwm.core.state as state_mod

    monkeypatch.setattr(state_mod, "cp", np)
    # CQ arena registration is opt-in under the stable default; enable it
    # explicitly because this test exercises the CQ-to-advection aliases.
    cfg_small = RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=1)
    cfg_large = RunConfig(**{**_TINY, "nx": 11, "ny": 7, "nz": 5},
                          moist=True, moist_cq=True, mp_physics=1)
    domains = (types.SimpleNamespace(run=cfg_small),
               types.SimpleNamespace(run=cfg_large))
    arena = state_mod.build_shared_scratch_arena(domains)
    expected_shape = (cfg_large.nz + 1, cfg_large.ny, cfg_large.nx)
    assert arena.slot_shapes["rk_ww"] == expected_shape

    small = state_mod.DomainState(cfg_small, scratch_arena=arena)
    large = state_mod.DomainState(cfg_large, scratch_arena=arena)
    small_ww = small.scratch(
        (cfg_small.nz + 1, cfg_small.ny, cfg_small.nx), "rk_ww")
    large_ww = large.scratch(expected_shape, "rk_ww")
    assert np.shares_memory(small_ww, large_ww)
    assert np.count_nonzero(large_ww) == 0
    small_ww.reshape(-1)[0] = np.float32(7.0)
    assert large_ww.reshape(-1)[0] == np.float32(7.0)

    # The three WRF cq faces add no physical arena backings: the acoustic
    # and standalone advection-only paths are mutually exclusive, and each
    # cq array is completely overwritten before its first stage read.
    aliases = pf.shared_scratch_arena_aliases(domains)
    assert aliases["acoustic_cqu"] == "adv_ru"
    assert aliases["acoustic_cqv"] == "adv_rv"
    assert aliases["acoustic_cqw"] == "adv_rw"
    for cq, adv in (("acoustic_cqu", "adv_ru"),
                    ("acoustic_cqv", "adv_rv"),
                    ("acoustic_cqw", "adv_rw")):
        cq_view = large.scratch(arena.slot_shapes[cq], cq)
        adv_view = large.scratch(arena.slot_shapes[adv], adv)
        assert np.shares_memory(cq_view, adv_view)

    default_a = state_mod.DomainState(cfg_small)
    default_b = state_mod.DomainState(cfg_small)
    a = default_a.scratch((2, 3), "unit_default")
    b = default_b.scratch((2, 3), "unit_default")
    assert not np.shares_memory(a, b)
    assert np.count_nonzero(a) == np.count_nonzero(b) == 0


def test_diff6_tendencies_alias_lifetime_safe_backings(monkeypatch):
    """Diff6-only uses one backing; every Smag path keeps x/y distinct."""
    import types

    import gpuwm.core.state as state_mod

    monkeypatch.setattr(state_mod, "cp", np)
    tiny = RunConfig(**_TINY, moist=True, mp_physics=10,
                     diff_6th_opt=2)
    domains = (types.SimpleNamespace(run=tiny),)
    shapes = pf.shared_scratch_arena_shapes(domains)
    aliases = pf.shared_scratch_arena_aliases(domains)
    assert {slot: aliases[slot]
            for slot in ("diff6_x", "diff6_y", "diff6_m")} == {
                "diff6_x": "diff6_z",
                "diff6_y": "diff6_z",
                "diff6_m": "diff6_z",
            }
    assert all(math.prod(shapes[slot]) <= math.prod(shapes["diff6_z"])
               for slot in ("diff6_x", "diff6_y", "diff6_m"))

    arena = state_mod.build_shared_scratch_arena(domains)
    z = arena.view(shapes["diff6_z"], "diff6_z")
    for slot in ("diff6_x", "diff6_y", "diff6_m"):
        assert np.shares_memory(arena.view(shapes[slot], slot), z)

    smag = RunConfig(**_TINY, moist=True, mp_physics=10, km_opt=4,
                     bl_pbl_physics=1)
    # A Smag-only domain has no z/m requests itself, but a shared arena may
    # acquire them from a different diff6 domain.  The global lifetime rule
    # must still keep the Smag x/y face buffers distinct.
    mixed_domains = (types.SimpleNamespace(run=smag), *domains)
    smag_aliases = pf.shared_scratch_arena_aliases(mixed_domains)
    assert smag_aliases["diff6_x"] == "diff6_z"
    assert smag_aliases["diff6_m"] == "diff6_z"
    assert "diff6_y" not in smag_aliases

    raw = tomllib.loads(CONFIG_4DOM.read_text(encoding="utf-8"))
    raw.pop("case_data")
    flagship = build_experiment(raw, source=str(CONFIG_4DOM))
    flagship_shapes = pf.shared_scratch_arena_shapes(flagship.domains)
    old_bytes = sum(
        4 * math.prod(flagship_shapes[slot])
        for slot in ("diff6_x", "diff6_y", "diff6_z", "diff6_m"))
    new_bytes = 4 * math.prod(flagship_shapes["diff6_z"])
    assert old_bytes == 283_915_200
    flagship_aliases = pf.shared_scratch_arena_aliases(flagship.domains)
    assert flagship_aliases["diff6_x"] == "diff6_z"
    assert flagship_aliases["diff6_m"] == "diff6_z"
    assert "diff6_y" not in flagship_aliases
    new_bytes += 4 * math.prod(flagship_shapes["diff6_y"])
    assert new_bytes == 142_677_600
    assert old_bytes - new_bytes == 141_237_600


def test_smag_coefficients_alias_acoustic_backings(monkeypatch):
    """Pre-RK K_m/K_h retire before acoustic alpha/gamma are prepared."""
    import types

    import gpuwm.core.state as state_mod

    monkeypatch.setattr(state_mod, "cp", np)
    tiny = RunConfig(**{**_TINY, "nx": 12, "ny": 10, "nz": 8},
                     moist=True, mp_physics=10, km_opt=4)
    domains = (types.SimpleNamespace(run=tiny),)
    shapes = pf.shared_scratch_arena_shapes(domains)
    aliases = pf.shared_scratch_arena_aliases(domains)
    assert aliases["smag_km"] == "acoustic_alpha"
    assert aliases["smag_kh"] == "acoustic_gamma"
    assert math.prod(shapes["smag_km"]) <= math.prod(
        shapes["acoustic_alpha"])
    assert math.prod(shapes["smag_kh"]) <= math.prod(
        shapes["acoustic_gamma"])

    arena = state_mod.build_shared_scratch_arena(domains)
    for slot, target in (("smag_km", "acoustic_alpha"),
                         ("smag_kh", "acoustic_gamma")):
        assert np.shares_memory(
            arena.view(shapes[slot], slot),
            arena.view(shapes[target], target))

    raw = tomllib.loads(CONFIG_4DOM.read_text(encoding="utf-8"))
    raw.pop("case_data")
    flagship = build_experiment(raw, source=str(CONFIG_4DOM))
    flagship_shapes = pf.shared_scratch_arena_shapes(flagship.domains)
    removed = sum(
        4 * math.prod(flagship_shapes[slot])
        for slot in ("smag_km", "smag_kh"))
    assert removed == 141_120_000


def _scan_scratch_tree(tree, rel):
    """(kind, payload, file, function) for every scratch(...) call site.

    Hardened per the review fix round: keyword-form slots
    (``scratch(shape, slot=...)`` / ``scratch(shape=..., slot=...)``) are
    inspected via ``Call.keywords``; a bare ``.scratch`` attribute LOAD
    that is not immediately called (method aliasing) and any
    ``getattr(x, "scratch")`` lookup are recorded as ``alias``/
    ``getattr`` sites so the completeness gate can reject them -- both
    were silent-skip bypasses (Fable F3 / shadow F4).
    """
    sites = []
    call_func_ids = {id(node.func) for node in ast.walk(tree)
                     if isinstance(node, ast.Call)}

    def slot_node(call):
        """(slot_node_or_None, is_scratch_call)."""
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "scratch":
            pos = 1
        elif isinstance(func, ast.Name) and func.id == "scratch":
            pos = 1
        elif isinstance(func, ast.Name) and func.id == "_lbc_scratch":
            pos = 2
        else:
            return None, False
        if len(call.args) > pos:
            return call.args[pos], True
        for kw in call.keywords:
            if kw.arg == "slot":
                return kw.value, True
        return None, True

    def visit(node, func):
        for child in ast.iter_child_nodes(node):
            name = func
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = child.name
            if isinstance(child, ast.Call):
                slot, is_scratch = slot_node(child)
                if is_scratch:
                    if slot is None:
                        sites.append(("no_slot", None, rel, func))
                    elif isinstance(slot, ast.Constant) and isinstance(
                            slot.value, str):
                        sites.append(("literal", slot.value, rel, func))
                    elif (isinstance(slot, ast.JoinedStr) and slot.values
                          and isinstance(slot.values[0], ast.Constant)
                          and isinstance(slot.values[0].value, str)):
                        sites.append(("prefix", slot.values[0].value,
                                      rel, func))
                    elif (isinstance(slot, ast.BinOp)
                          and isinstance(slot.left, ast.Constant)
                          and isinstance(slot.left.value, str)):
                        sites.append(("prefix", slot.left.value, rel, func))
                    else:
                        sites.append(("variable", None, rel, func))
                if (isinstance(child.func, ast.Name)
                        and child.func.id == "getattr"
                        and len(child.args) >= 2
                        and isinstance(child.args[1], ast.Constant)
                        and child.args[1].value == "scratch"):
                    sites.append(("getattr", None, rel, func))
            if (isinstance(child, ast.Attribute) and child.attr == "scratch"
                    and id(child) not in call_func_ids):
                # `sc = state.scratch` style method alias: the later
                # calls are invisible to the scanner, so the alias itself
                # is the violation.
                sites.append(("alias", None, rel, func))
            visit(child, name)

    visit(tree, "<module>")
    return sites


def _scratch_call_sites():
    sites = []
    for path in sorted((ROOT / "gpuwm").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        sites += _scan_scratch_tree(tree, path.relative_to(ROOT).as_posix())
    return sites


def test_every_scratch_call_site_is_classified(d01_cfg):
    """The plan's completeness gate: every ``scratch(...)`` call site in
    gpuwm/ resolves against the static registry -- literal slots must be
    registry names (or F4 manifest names for ``nest_*``), dynamic slots
    must use a registered prefix family, and variable-slot sites are
    pinned to an explicit allowlist.  An unclassified slot is an error."""
    known: set[str] = set()
    for cfg in (d01_cfg,
                RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=1,
                          open_x=True, open_y=True, khdif=1.0, kvdif=1.0,
                          emdiv=0.01),
                RunConfig(**_TINY, moist=True, mp_physics=6),
                RunConfig(**_TINY, moist=True, mp_physics=8),
                RunConfig(**_TINY, moist=True, mp_physics=18),
                RunConfig(**_TINY, sf_sfclay_physics=1),
                RunConfig(**_TINY, nwp_diagnostics=1)):
        known |= set(pf.scratch_slot_registry(cfg, n_lbc_intervals=2))
    exp = load_experiment_case(CONFIG_4DOM)[0]
    for dc in exp.domains:
        known |= set(pf.scratch_slot_registry(dc.run, n_lbc_intervals=2))
    manifest_names: set[str] = set()
    for slots in pf.nest_allocation_manifest(exp).values():
        manifest_names |= set(slots)

    known_prefixes = {"cu_", "smag_r", "lbc_weights_",
                      "lbc_old_mup_frame_", "lbc_relax_",
                      # spec-zone ring-guard snapshot family: registry
                      # shapes from microphysics.spec_zone_ring_save_slots,
                      # lifetime row "mp_ring_save_*" (excluded_unproven),
                      # restart REBUILT_SCRATCH_PREFIXES entry.
                      "mp_ring_save_"}
    allowed_variable_sites = {
        ("gpuwm/core/dycore.py", "add_smag2d_tendencies"),
        ("gpuwm/core/dycore.py", "_compute_wrf_smag_tendencies"),
        ("gpuwm/core/dycore.py", "prepare_fixed_tendencies"),
        ("gpuwm/core/dycore.py", "add_fixed_dry_tendencies"),
        ("gpuwm/core/dycore.py", "apply_diff6"),
        ("gpuwm/core/diffusion.py", "add_diffusion_tendencies"),
        ("gpuwm/core/physics.py", "__init__"),
        ("gpuwm/io/restart.py", "_apply_validated_restart"),
        ("gpuwm/io/restart.py", "_restore_driver"),
        ("gpuwm/ingest/lateral_bc.py", "_lbc_scratch"),
        ("gpuwm/ingest/lateral_bc.py", "_resident_weights"),
        ("gpuwm/ingest/lateral_bc.py", "attach_lateral_boundaries"),
        ("gpuwm/ingest/lateral_bc.py", "attach_streaming_lateral_boundaries"),
        ("gpuwm/core/preflight.py", "run_alloc_preflight"),
        ("gpuwm/core/nest.py", "_scratch"),
        # MYNN draws its whole declared workspace in two loops over the same
        # shape functions the registry calls, so the slot expressions are
        # variables by construction.  Allowlisting them here would be a hole
        # on its own; what closes it is
        # tests/test_mynn_pbl_scratch.py::test_the_registry_prices_exactly_
        # the_slots_the_solver_asks_for, which runs a real MYNN forecast with
        # DomainState.scratch instrumented and requires the requested slot
        # set to equal preflight.mynn_pbl_scratch_slots(cfg) exactly -- both
        # directions, so neither an unpriced slot nor a stale registry row
        # survives.
        ("gpuwm/core/mynn_pbl_scratch.py", "from_state"),
        ("gpuwm/core/mynn_pbl_runtime.py", "mynn_pbl_step"),
    }
    # The one sanctioned getattr(state, "scratch") lookup: lateral_bc's
    # duck-type guard, whose resulting Name call the scanner classifies.
    allowed_getattr_sites = {
        ("gpuwm/ingest/lateral_bc.py", "_lbc_scratch"),
    }

    sites = _scratch_call_sites()
    assert sites, "AST scan found no scratch call sites -- scanner broken"
    problems = []
    seen_variable_sites = set()
    for kind, payload, rel, func in sites:
        if kind == "literal":
            if payload.startswith("nest_"):
                if payload not in manifest_names:
                    problems.append(f"{rel}::{func}: nest slot {payload!r} "
                                    "is not in the F4 allocation manifest")
            elif payload not in known:
                problems.append(f"{rel}::{func}: slot {payload!r} is not in "
                                "the scratch registry")
        elif kind == "prefix":
            if payload not in known_prefixes:
                problems.append(f"{rel}::{func}: dynamic slot prefix "
                                f"{payload!r} is not a registered family")
        elif kind == "variable":
            seen_variable_sites.add((rel, func))
            if (rel, func) not in allowed_variable_sites:
                problems.append(f"{rel}::{func}: variable slot expression "
                                "is not in the pinned allowlist")
        elif kind == "getattr":
            if (rel, func) not in allowed_getattr_sites:
                problems.append(f"{rel}::{func}: getattr(..., 'scratch') "
                                "lookup escapes the completeness gate")
        else:  # "alias" / "no_slot": never legitimate
            problems.append(f"{rel}::{func}: {kind} scratch usage escapes "
                            "the completeness gate")
    assert not problems, "\n".join(problems)
    # The allowlist may not silently rot either.
    assert seen_variable_sites == allowed_variable_sites


def test_experimental_thompson_scratch_registry_is_complete():
    cfg = RunConfig(**_TINY, moist=True, mp_physics=8)
    slots = pf.scratch_slot_registry(cfg)
    mass = (cfg.nz, cfg.ny, cfg.nx)
    surface = (cfg.ny, cfg.nx)
    assert {
        "mp_th": mass,
        "mp_pii": mass,
        "mp_dz8w": mass,
        "mp_z8w": (cfg.nz + 1, cfg.ny, cfg.nx),
        "mp_thompson_temperature": mass,
        "mp_thompson_frozen_reference_density": mass,
        "mp_thompson_frozen_reference_temperature": mass,
        "mp_thompson_rain_reference_density": mass,
        "mp_thompson_snow_melt_marker": mass,
        "mp_thompson_graupel_melt_marker": mass,
        "mp_thompson_snow_velocity_boost": mass,
        "mp_thompson_graupel_number_shadow": mass,
        "mp_rainnc": surface,
        "mp_rainncv": surface,
        "mp_snownc": surface,
        "mp_snowncv": surface,
        "mp_graupelnc": surface,
        "mp_graupelncv": surface,
        "mp_sr": surface,
        "refl_t": mass,
        "refl_10cm": mass,
    }.items() <= slots.items()
    for slot in (
            "mp_thompson_temperature",
            "mp_thompson_frozen_reference_density",
            "mp_thompson_frozen_reference_temperature",
            "mp_thompson_rain_reference_density",
            "mp_thompson_snow_melt_marker",
            "mp_thompson_graupel_melt_marker",
            "mp_thompson_snow_velocity_boost",
            "mp_thompson_graupel_number_shadow"):
        assert pf.scratch_slot_uses_arena(slot)


def test_scratch_scanner_catches_the_review_bypasses():
    """Regression fixtures: the exact bypass constructions from the p5t11
    reviews (keyword-form slots, one-positional + keyword slot, method
    aliasing, getattr lookup) must be seen -- silent skips were the F3/F4
    MAJOR."""
    def kinds(src):
        return [(kind, payload) for kind, payload, _, _ in
                _scan_scratch_tree(ast.parse(src), "synthetic.py")]

    # Shadow F4 construction: both-keyword form.
    assert kinds("state.scratch(shape=(nz, ny, nx), "
                 "slot='unregistered_resident')") == [
        ("literal", "unregistered_resident")]
    # One positional + keyword slot.
    assert kinds("state.scratch((nz, ny, nx), slot='nest_bogus')") == [
        ("literal", "nest_bogus")]
    # Keyword f-string still classifies as a prefix family.
    assert kinds("state.scratch((2, w), slot=f'lbc_weights_{n}')") == [
        ("prefix", "lbc_weights_")]
    # Fable F3(2): method alias under another name -- the alias itself
    # is flagged even though the later call is unrecognizable.
    assert ("alias", None) in kinds("sc = state.scratch\nsc((1,), 'x')")
    # Fable F3(3): getattr lookup, stored or immediately called.
    assert ("getattr", None) in kinds(
        "getattr(state, 'scratch')((1,), 'x')")
    assert ("getattr", None) in kinds(
        "f = getattr(state, 'scratch', None)")
    # A slot the scanner cannot identify at all is a finding, not a skip.
    assert kinds("state.scratch((1,))") == [("no_slot", None)]
    # Ordinary calls stay classified exactly as before.
    assert kinds("state.scratch((1,), 'rk_ww')") == [("literal", "rk_ww")]
    assert kinds("state.scratch(f.shape, slotvar)") == [("variable", None)]


# ---------------------------------------------------------------------------
# (d) LBC residents + the F4 nest allocation manifest
# ---------------------------------------------------------------------------

def test_lbc_interval_values_hand_check(d01_cfg):
    """Independent hand arithmetic for one interval's side tables
    (W=5, nz=49, ny=200, nx=250; value+tendency, 4 sides per field)."""
    per_field = {
        "u": 2 * (2 * 49 * 200 * 5 + 2 * 49 * 5 * 251),
        "v": 2 * (2 * 49 * 201 * 5 + 2 * 49 * 5 * 250),
        "theta": 2 * (2 * 49 * 200 * 5 + 2 * 49 * 5 * 250),
        "qv": 2 * (2 * 49 * 200 * 5 + 2 * 49 * 5 * 250),
        "phi": 2 * (2 * 50 * 200 * 5 + 2 * 50 * 5 * 250),
        "mu": 2 * (2 * 1 * 200 * 5 + 2 * 1 * 5 * 250),
    }
    assert pf.lbc_interval_values(d01_cfg) == sum(per_field.values())
    assert pf.lbc_interval_values(d01_cfg) == 2224960
    assert pf.lbc_intervals(43200.0, 21600.0) == 2
    assert pf.lbc_intervals(43201.0, 21600.0) == 3


def test_nest_field_kinds_by_scheme(exp4):
    dry = RunConfig(**_TINY)
    assert pf.nest_field_kinds(dry) == ("u", "v", "w", "t", "ph", "mu")
    kessler = RunConfig(**_TINY, moist=True, mp_physics=1)
    assert pf.nest_field_kinds(kessler) == (
        "u", "v", "w", "t", "ph", "mu", "qv", "qc", "qr")
    # Morrison: all active species incl. the scalar numbers; nc excluded
    # (no advection copy -- state.py has no nc0).
    assert pf.nest_field_kinds(exp4.domain(2).run) == (
        "u", "v", "w", "t", "ph", "mu", "qv", "qc", "qr",
        "qi", "qs", "qg", "nr", "ni", "ns", "ng")


def test_nest_allocation_manifest_inventory(exp4):
    manifest = pf.nest_allocation_manifest(exp4)
    assert sorted(manifest) == [2, 3, 4]  # root never registers nest slots
    for slots in manifest.values():
        # 16 kinds x 4 sides x (value + tendency), six geometry arrays
        # per three staggers, plus simultaneously live arena-audited parent
        # and child full fields.
        assert len(slots) == 16 * 4 * 2 + 3 * 6 + 2 == 148
    d02 = manifest[2]
    # Rolling tables: WRF Registry naming/layout (u_bxs/u_btxs style).
    assert d02["nest_u_bxs"] == (49, 400, 5)
    assert d02["nest_u_btxs"] == (49, 400, 5)
    assert d02["nest_u_btys"] == (49, 5, 501)
    assert d02["nest_v_bxs"] == (49, 401, 5)
    assert d02["nest_w_bxs"] == (50, 400, 5)
    assert d02["nest_ph_bye"] == (50, 5, 500)
    assert d02["nest_mu_bye"] == (1, 5, 500)
    assert d02["nest_ng_bxe"] == (49, 400, 5)
    # F16 retires every donor strip.  One full-parent field is borrowed
    # from the shared force-only arena; d02's parent d01 has 50*200*250
    # full-level w values, the largest parent field on that edge.
    assert not any("donor" in name for name in d02)
    assert d02["nest_parent_field"] == (50 * 200 * 250,)
    assert manifest[3]["nest_parent_field"] == (50 * 400 * 500,)
    assert manifest[4]["nest_parent_field"] == (50 * 501 * 501,)
    # T10 device_tables registry: ci/ip/cj/jp int32 maps and ratio-length
    # xig/xjg float32 coefficients, independently stored per stagger.
    assert d02["nest_sint_ci_m"] == (500,)
    assert d02["nest_sint_ip_x"] == (501,)
    assert d02["nest_sint_cj_y"] == (401,)
    assert d02["nest_sint_jp_y"] == (401,)
    assert d02["nest_sint_xig_m"] == (4,)
    assert manifest[4]["nest_sint_xjg_m"] == (3,)
    dtypes = pf.nest_slot_dtypes(exp4.domain(2), exp4.spec_bdy_width,
                                 exp4.domain(1))
    assert dtypes["nest_sint_ci_m"] == "int32"
    assert dtypes["nest_sint_jp_y"] == "int32"
    assert dtypes["nest_sint_xig_m"] == "float32"
    assert dtypes["nest_u_bxs"] == "float32"

    # Logical-request footprint pins.  The simultaneously live full-parent
    # and full-child fields are counted here even though the physical arena
    # aliases them to distinct dead RK backings when capacities permit.
    totals = {gid: sum(4 * math.prod(shape) for shape in slots.values())
              for gid, slots in manifest.items()}
    assert totals == {2: 103165552, 3: 149390256, 4: 193084928}
    assert sum(totals.values()) < 450 * 1024 ** 2


# ---------------------------------------------------------------------------
# (e) RRTMGP workspace/chunk formula
# ---------------------------------------------------------------------------

def test_gas_table_meta_and_default_chunk():
    meta = pf._gas_table_meta()
    assert meta["ngpt_lw"] == 256 and meta["ngpt_sw"] == 224
    from gpuwm.core.rrtmgp import RRTMGPRadiation
    default = {f.name: f.default
               for f in dataclasses.fields(RRTMGPRadiation)}["column_chunk"]
    assert pf.DEFAULT_COLUMN_CHUNK == default == 3125
    assert pf._workspace_total_bytes(49, default) == 1025700000


def test_estimate_uses_experiment_column_chunk(exp4):
    configured = dataclasses.replace(exp4, column_chunk=6250)
    estimate = pf.estimate_experiment(configured)
    assert estimate.column_chunk == 6250
    assert estimate.workspace_bytes == pf._workspace_total_bytes(49, 6250)


def test_rrtmgp_chunk_loops_and_mcica_seed_are_column_local():
    """Static pin for A-1's no-chunk-size-arithmetic proof.

    Both solver loops may use their absolute ``start`` only to construct the
    input/output slice.  McICA selects the same column's bottom pressures and
    never folds a local/global column number or chunk size into its seeds.
    """
    import inspect
    import textwrap

    from gpuwm.core.rrtmgp import RRTMGPRadiation

    source = textwrap.dedent(inspect.getsource(RRTMGPRadiation.__call__))
    tree = ast.parse(source)
    loops = [node for node in ast.walk(tree)
             if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name)
             and node.target.id == "start"]
    assert len(loops) == 2
    for loop in loops:
        loaded_start = [node for node in ast.walk(loop)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id == "start"]
        assert len(loaded_start) == 2  # slice start and slice stop only
        loop_source = ast.get_source_segment(source, loop)
        assert not any(token in loop_source for token in (
            "sum(", "mean(", "cumsum(", "reduce("))
        assert loop_source.count("_prepare_above_model_chunk(") == 1
        assert loop_source.count("columns=sl") == 1
    assert source.count("_model_flux_interfaces(") == 4
    assert "lw_up[sl] = _model_flux_interfaces(" in source
    assert "sw_up[sl] = cp.where" in source
    assert "sw_dn[sl] = cp.where" in source

    kernel = (ROOT / "gpuwm" / "core" / "kernels" /
              "rrtmgp_mcica.cu").read_text(encoding="utf-8")
    seed_start = kernel.index("unsigned int s1, s2, s3, s4;")
    seed_end = kernel.index("for (int g = 0; g < ngpt; ++g)")
    seed = kernel[seed_start:seed_end]
    assert "play[col * nlay + n]" in seed
    assert "frac * 1.0e9" in seed
    assert "permuteseed" in seed
    assert "blockIdx" not in seed and "chunk" not in seed


def test_workspace_is_the_phase_maximum_simultaneous_set():
    """Shadow F2 fix: the workspace bound is the max over the four solver
    phases' EXACT live sets (col_dry included; post-mask-delete arrays
    not substituted into the mask phase).  WRF's 100-hPa cap adds 25 LW
    and one SW layer; LW RTE is therefore the new phase maximum."""
    phases = pf.rrtmgp_workspace_phases(49, 12500)

    def total(items):
        return sum(math.prod(shape) * size
                   for shape, size in items.values())

    assert {name: total(items) for name, items in phases.items()} == {
        "lw_optics": 2386500000,
        "lw_rte": 4102800000,
        "sw_optics": 3097500000,
        "sw_rte": 2990050000,
    }
    # Every phase carries col_dry (rrtmgp.py:1615, retained by both the
    # gas and finalized optics results -- one shared array).
    assert phases["lw_optics"]["col_dry"] == ((12500, 74), 4)
    assert phases["lw_rte"]["col_dry"] == ((12500, 74), 4)
    assert phases["sw_optics"]["col_dry"] == ((12500, 50), 4)
    assert phases["sw_rte"]["col_dry"] == ((12500, 50), 4)
    # The later sw_rte phase enumerates mu0 + the three flux arrays.
    assert phases["sw_rte"]["mu0"] == ((12500, 50), 4)
    assert phases["sw_rte"]["flux_dir"] == ((12500, 51), 4)
    # The albedo/incidence arrays exist only after the mask is deleted.
    assert "albedo_gpt" not in phases["sw_optics"]
    assert "mcica_mask" not in phases["sw_rte"]

    full = pf._workspace_total_bytes(49, 12500)
    assert full == 4102800000
    assert pf._workspace_total_bytes(49, 6250) * 2 == full
    assert pf._workspace_total_bytes(49, 3125) * 4 == full
    shapes = pf.rrtmgp_workspace_shapes(49, 12500)
    assert all(name.startswith("lw_rte/") for name in shapes)
    assert shapes["lw_rte/gas_tau"] == ((12500, 74, 256), 4)
    assert shapes["lw_rte/lev_source"] == ((12500, 75, 256), 4)
    toa_column = pf.rrtmgp_workspace_phases(49, 2, p_top=0.0)
    assert toa_column["lw_rte"]["gas_tau"] == ((2, 49, 256), 4)
    assert toa_column["sw_rte"]["gas_tau"] == ((2, 49, 224), 4)


def test_shared_rrtmgp_workspace_is_one_real_allocation_with_full_audit():
    import inspect

    from gpuwm.core.model import SharedRRTMGPChunkWorkspace
    from gpuwm.core.rrtmgp import (RRTMGP_WORKSPACE_LIFETIME_AUDIT,
                                   RRTMGPRadiation)

    layouts = pf.rrtmgp_workspace_phases(3, 2)
    workspace = SharedRRTMGPChunkWorkspace(
        nz=3, column_chunk=2, _array_module=np,
        _phase_layouts_input=layouts)
    assert workspace.p_top == 10000.0
    assert workspace.nbytes == pf._workspace_total_bytes(3, 2)
    for phase, items in layouts.items():
        views = workspace.phase(phase, 2)
        assert set(views) == set(items)
        assert all(np.shares_memory(value, workspace.storage)
                   for value in views.values())
        assert set(RRTMGP_WORKSPACE_LIFETIME_AUDIT[phase]) == set(items)
        assert all(RRTMGP_WORKSPACE_LIFETIME_AUDIT[phase].values())

    # Common optics live values retain the same address in the immediately
    # following RTE layout; only dead tail storage is repurposed.
    for kind in ("lw", "sw"):
        optics = workspace.phase(f"{kind}_optics", 2)
        rte = workspace.phase(f"{kind}_rte", 2)
        for name in set(optics) & set(rte):
            assert optics[name].ctypes.data == rte[name].ctypes.data

    # Distinct per-domain adapters consume the same allocated workspace, not
    # merely equal capacity tokens.
    first = object.__new__(RRTMGPRadiation)
    second = object.__new__(RRTMGPRadiation)
    first.chunk_workspace = second.chunk_workspace = workspace
    assert first.chunk_workspace.storage is second.chunk_workspace.storage
    driver_source = inspect.getsource(RRTMGPRadiation.__call__)
    for phase in layouts:
        assert f'workspace.phase("{phase}"' in driver_source
    for producer in ("_gas_optics", "_cloud_optics", "_mcica_cloud_masks",
                     "_finalize_cloud_optics", "_planck_sources",
                     "_lw_rte", "_sw_rte"):
        assert producer in driver_source


def test_rrtmgp_column_transients(d01_cfg):
    cols = pf.rrtmgp_column_shapes(d01_cfg)
    ncol = d01_cfg.ny * d01_cfg.nx
    chunk = pf.DEFAULT_COLUMN_CHUNK
    assert cols["columns/play"] == ((ncol, 49), 4)
    assert cols["columns/plev"] == ((ncol, 50), 4)
    assert cols["columns/effs"] == ((ncol, 49), 4)  # Morrison extras
    assert cols["columns/metadata_jt"] == ((chunk, 74), 4)
    assert cols["columns/upper_peak_play"] == ((chunk, 74), 4)
    assert cols["columns/upper_peak_plev"] == ((chunk, 75), 4)
    assert pf.rrtmgp_column_shapes(
        d01_cfg, column_chunk=17)["columns/metadata_jt"] == ((17, 74), 4)
    assert pf.rrtmgp_column_shapes(RunConfig(**_TINY)) == {}


# ---------------------------------------------------------------------------
# Estimates: itemization, shared counting, golden pins, d01 calibration
# ---------------------------------------------------------------------------

def test_estimate_domain_itemization_pins(exp1):
    est = pf.estimate_experiment(exp1)
    (d01,) = est.domains
    assert not est.uses_shared_scratch_arena
    assert not est.uses_shared_dycore_state_workspace
    assert est.scratch_arena_bytes == est.scratch_arena_saved_bytes == 0
    assert est.dycore_state_workspace_bytes == est.dycore_state_saved_bytes == 0
    by_cat = {c: d01.category_bytes(c) for c in
              ("state", "physics", "scratch", "lbc", "nest", "transient")}
    # Stable moist_cq=False omits the three float32 CQ faces:
    # 4 * (49*200*251 + 49*201*250 + 50*200*250) = 29,688,200 B.
    # Ring-guard note: the spec-zone microphysics exclusion adds its
    # mp_ring_save_* snapshot family to a specified mp=10 domain --
    # 17 volume slots (16 mutated fields + refl_10cm stash) x
    # 4*49*(2*250 + 2*198) = 175,616 B, plus 7 surface slots x
    # 4*(2*250 + 2*198) = 3,584 B: 2,985,472 + 25,088 = 3,010,560 B on
    # top of the previous 564,250,212-B scratch pin.
    assert by_cat == {
        "state": 563557756,
        "physics": 275706760,
        # KF hold + expiry mask + ring-guard saves, plus the v1.1
        # co-located vertical-CFL reduction field: one extra FP32 word in
        # each of the 256 `integration_health_partial` blocks and in the
        # single `health_final` block, 4 * (256 + 1) = 1,028 B.  Batched YSU
        # validation adds one four-byte scratch status word.
        "scratch": 567286380,
        "lbc": 67091504,
        "nest": 0,
        "transient": 441262500,
    }
    assert d01.resident_bytes == sum(
        v for c, v in by_cat.items() if c != "transient")
    assert d01.resident_bytes == 1473642400
    assert est.resident_bytes == d01.resident_bytes + est.k_tables_bytes
    assert d01.transient_bytes == 441262500


def test_estimate_experiment_shared_counting(est4, exp4):
    # k-distribution tables counted ONCE (lru_cache-shared,
    # rrtmgp.py:324/:436), while audited scratch is one per-slot maximum.
    assert est4.k_tables_bytes == pf.k_distribution_bytes() == 23763712
    assert est4.uses_shared_scratch_arena
    assert est4.scratch_arena_request_bytes == sum(
        d.arena_scratch_bytes for d in est4.domains)
    assert est4.scratch_arena_bytes == pf.shared_scratch_arena_bytes(
        exp4.domains)
    assert est4.uses_shared_dycore_state_workspace
    assert est4.dycore_state_request_bytes == sum(
        d.rebuilt_state_bytes for d in est4.domains)
    assert est4.dycore_state_workspace_bytes == (
        pf.shared_dycore_state_workspace_bytes(exp4.domains))
    assert est4.resident_bytes == (
        sum(d.resident_bytes for d in est4.domains)
        - est4.scratch_arena_saved_bytes
        - est4.dycore_state_saved_bytes + est4.k_tables_bytes)
    # Step transients take the max over sequentially stepping domains.
    assert est4.transient_peak_bytes == max(
        d.transient_bytes for d in est4.domains)
    assert est4.transient_peak_bytes == est4.domains[-1].transient_bytes
    # ONE shared chunk workspace for all four domains (section E policy),
    # sized by the CASE's configured chunk, not the library default.
    assert exp4.column_chunk == 6250 != pf.DEFAULT_COLUMN_CHUNK
    assert est4.workspace_bytes == pf._workspace_total_bytes(
        49, exp4.column_chunk)


def test_estimate_4dom_golden_pins(exp4, est4):
    per_domain = {d.grid_id: d.resident_bytes for d in est4.domains}
    # The production MP18 authority enables moist-CQ on every domain. Direct
    # MUDF storage still removes the obsolete acoustic_muprev mass plane.
    # Assembly merge (verification lineage): the ring-guard mp_ring_save_*
    # snapshot slots add exactly 3,010,560 / 6,034,560 / 6,720,000 /
    # 8,050,560 B on d01--d04 (sum 23,815,680 B, the ring lane's ledgered
    # 4-domain total) on top of the ports-branch pins; both components
    # byte-derived on their certified branches.
    # v1.1 hygiene merge: the co-located vertical-CFL reduction adds one
    # FP32 word to each of the 256 `integration_health_partial` blocks and
    # to `health_final`, so every domain gains exactly 4 * (256 + 1) =
    # 1,028 B over the ring-lane pins.  Constant per domain because the
    # health reduction's block count does not scale with the grid.  The
    # batched YSU validator adds one four-byte status word per domain.
    assert per_domain == {1: 1503330600, 2: 5403118992,
                          3: 6799612696, 4: 9728899968}
    nest = {d.grid_id: d.category_bytes("nest") for d in est4.domains}
    assert nest == {1: 0, 2: 103165552, 3: 149390256, 4: 193084928}
    # The post-CQ request includes both simultaneously live nested-force
    # full-field slots.  The shared physical arena still aliases them to
    # distinct dead RK backings when those capacities fit.
    assert est4.scratch_arena_request_bytes == 9471818140
    # Every Smag path requires distinct horizontal face staging; x/m reuse z
    # while y remains independent.  The pre-RK Smag K pair then borrows the
    # later acoustic coefficient backings.  These remove 141,237,600 B and
    # 141,120,000 B of exact physical allocation.
    assert est4.scratch_arena_bytes == 3315315836
    # 9,471,818,140 requested - 3,315,315,836 physical = 6,156,502,304 B.
    assert est4.scratch_arena_saved_bytes == 6156502304
    assert est4.dycore_state_request_bytes == 4930458300
    assert est4.dycore_state_workspace_bytes == 2061345600
    assert est4.dycore_state_saved_bytes == 2869112700
    # Ring snapshot slots are resident (arena-excluded): the ports-branch
    # residency plus the exact 23,815,680-B ring total.
    assert est4.resident_bytes == 14433110964
    # The case configures column_chunk = 6250 (byte-identical to 3125,
    # 33% faster per radiation call); the 3125 numbers stay pinned in the
    # ladder below, so the trade this bought is on the record both ways.
    assert est4.workspace_bytes == 2051400000
    assert est4.transient_peak_bytes == 3182840000
    assert est4.subtotal_bytes == 19667350964
    assert est4.alloc_estimate_bytes == math.ceil(
        1.15 * est4.subtotal_bytes) == 22617453609
    # Chunk ladder after arena sharing and physics-persistent reclamation.
    # The 1024-descriptor health-slot registration adds 49,168 B/domain to
    # the audited scratch; the pins below are computed on the merged tree.
    ladder = {chunk: pf.estimate_experiment(
        exp4, column_chunk=chunk).alloc_estimate_bytes
        for chunk in (6250, 3125, 1562, 256)}
    # Every rung carries the ring lane's ceil(1.15 x 23,815,680) =
    # 27,388,032 B on top of the ports-branch ladder.
    assert ladder == {6250: 22617453609, 3125: 21421913609,
                      1562: 20823952323, 256: 20324312246}


def test_d01_calibration_bounds_measured_fixture(exp1):
    """The pre-reclamation d01 measurement remains below the full estimate.

    Its persistent-used value is no longer a tight residency calibration:
    that run deliberately retained last_ysu, a composed stack, and copied
    diagnostics which this lane removes.
    """
    est = pf.estimate_experiment(exp1)
    measured = pf.CAL_D01_POOL_USED_PEAK_BYTES
    # Enforced bound: measured <= estimate.
    assert est.alloc_estimate_bytes >= measured
    # CQ-off removes 29,688,200 B: 1,451,294,432 - 29,688,200 =
    # 1,421,606,232 B, so the old fixture is now about 11% above residency.
    ratio = est.domains[0].resident_bytes / measured
    assert 0.93 <= ratio <= 0.94


def test_calibration_constants_pin_the_measurement_record():
    """CONSISTENCY pins, not validation (shadow F3 / Fable F5): these
    re-state the two controller measurement records -- the d01 run
    fixture (n0-preflight-baseline.log) and the N0 allocation probe
    (n0-alloc-probe-r2.json) -- so any silent constant edit is visible.
    The tier-2/3 model built on them is provisional reserve POLICY for
    controller ratification; nothing here can validate it against
    independent evidence."""
    assert pf.CAL_WDDM_FREE_BYTES == int(30.27 * GIB)
    assert pf.CAL_WDDM_TOTAL_BYTES == int(31.84 * GIB)
    assert pf.CAL_D01_POOL_USED_PEAK_BYTES == int(1.47 * GIB)
    assert pf.CAL_D01_POOL_HELD_BYTES == int(5.52 * GIB)
    assert pf.CAL_D01_DEVICE_FOOTPRINT_BYTES == int(11.24 * GIB)
    assert pf.CAL_FIXTURE_OVERHEAD_BYTES == int(11.24 * GIB) - int(5.52 * GIB)
    assert pf.CAL_D01_POOL_RETENTION_BYTES == (int(5.52 * GIB)
                                               - int(1.47 * GIB))
    assert pf.ALLOCATOR_HEADROOM == 1.15
    # N0 probe record: the fixture's 5.72 GiB memGetInfo gap was 12 h-run
    # drift -- a fresh allocation-only process measures 1.39 GiB, and
    # allocation-time pool retention is nil (16 MB).
    assert pf.PROBE_DEVICE_OVERHEAD_BYTES == 1489949696
    assert (pf.PROBE_POOL_HELD_BYTES
            - pf.PROBE_POOL_USED_PEAK_BYTES) == 16154112
    assert pf.PROBE_DEVICE_OVERHEAD_BYTES == (
        pf.PROBE_DEVICE_FOOTPRINT_BYTES - pf.PROBE_POOL_HELD_BYTES)


def test_tier_projection_algebra_is_consistent(exp1):
    """The tier-2/3 projections are labels over the estimate, pinned as
    ALGEBRAIC IDENTITIES (the old ">= fixture" assertions were
    tautologies -- the residual/overhead terms embed the same fixture
    numbers they were claimed to bound; shadow F3)."""
    est = pf.estimate_experiment(exp1)
    assert est.held_projection_bytes == (
        est.alloc_estimate_bytes + est.retention_residual_bytes)
    assert est.footprint_projection_bytes == (
        est.held_projection_bytes + pf.PROBE_DEVICE_OVERHEAD_BYTES)
    assert est.retention_residual_bytes == \
        pf.pool_retention_residual_bytes()


# ---------------------------------------------------------------------------
# Reserve policy + the F11 enforced chain
# ---------------------------------------------------------------------------

def test_reserve_policy_split_proposals():
    """The two reserve proposals, split by gate (PENDING controller
    ratification at N0; instruction #5 of the fix round): the N0
    allocation gate carries only the probe-measured fresh-process
    overhead + external margin (alloc-time retention measured nil); the
    N5/N6 run gates add the fixture-calibrated run-churn residual."""
    n0 = pf.ReservePolicy.n0_alloc()
    assert n0.retention_residual_bytes == 0
    assert n0.device_overhead_bytes == pf.PROBE_DEVICE_OVERHEAD_BYTES
    assert n0.reserve_bytes == (pf.PROBE_DEVICE_OVERHEAD_BYTES
                                + pf.EXTERNAL_MARGIN_BYTES) == 2026820608

    run = pf.ReservePolicy.run_time()
    assert run.retention_residual_bytes == \
        pf.pool_retention_residual_bytes()
    assert run.reserve_bytes == n0.reserve_bytes + \
        pf.pool_retention_residual_bytes()
    assert run.reserve_bytes == 4959159922

    flat = pf.ReservePolicy.flat(2 * GIB)
    assert flat.reserve_bytes == 2 * GIB
    assert flat.budget_bytes(pf.PROBE_FREE_BYTES) == (
        pf.PROBE_FREE_BYTES - 2 * GIB)

    # The run-churn residual: fixture held minus the d01 alloc-estimate
    # basis (measured used + phase-max workspace, with headroom),
    # clamped at zero -- calibration algebra, pinned.
    basis = math.ceil(pf.ALLOCATOR_HEADROOM * (
        pf.CAL_D01_POOL_USED_PEAK_BYTES
        + pf._workspace_total_bytes(49, pf.DEFAULT_COLUMN_CHUNK)))
    assert pf.pool_retention_residual_bytes() == max(
        0, pf.CAL_D01_POOL_HELD_BYTES - basis) == 2932339314


def test_n0_probe_projection_flags_stale_calibration_after_exact_aliases(
        exp4, est4):
    """The old probe cannot certify the post-alias measured-bound gate.

    Its pool-used projection loses the exact physical diff6 and Smag bytes,
    while the proportional estimator loses 1.15 times those amounts.
    Historically the projection sat 26,185,543 B ABOVE the estimate and
    the measured-bound leg exposed the staleness.  The ring-guard
    mp_ring_save_* family then grew the audited scratch by 23,815,680 B
    across the four domains (x1.15 headroom = 27,388,032 B of estimate),
    which consumed that margin: composed with the ports-branch growth
    (health descriptors, nested-force slots) the stale projection now sits
    1,316,725 B BELOW the estimate and the measured-bound leg reads True.  The
    algebra stays pinned exactly; the operational consequence is that the
    retained pre-alias receipt can no longer flag its own staleness
    through this leg -- the next N0 certification MUST take a fresh probe
    receipt (PROBE_POOL_USED_PEAK_BYTES) rather than trust this
    projection.
    """
    # ``PROBE_POOL_USED_PEAK_BYTES`` is a receipt taken at the LIBRARY default
    # chunk, so it must be projected against the default-chunk estimate.
    # Comparing it to the case's configured 6250 estimate would flip
    # ``alloc_measured_le_estimate`` to True on 1.03 GB of workspace the probe
    # never allocated -- concealing exactly the staleness this test exposes.
    est = pf.estimate_experiment(exp4, column_chunk=pf.DEFAULT_COLUMN_CHUNK)
    diff6_alias_saved = 141_237_600
    projected_used = (pf.PROBE_POOL_USED_PEAK_BYTES
                      - est.dycore_state_saved_bytes
                      - (3035550000 - est.workspace_bytes)
                      - diff6_alias_saved - 141_120_000)
    assert projected_used - est.alloc_estimate_bytes == -1_316_725
    legs = pf.evaluate_alloc_gates(
        measured_used_bytes=projected_used,
        estimate_bytes=est.alloc_estimate_bytes,
        measured_free_bytes=pf.PROBE_FREE_BYTES,
        reserve=pf.ReservePolicy.flat(2 * GIB))
    assert legs == {"alloc_fits_wddm_budget": True,
                    "alloc_measured_le_estimate": True,
                    "alloc_estimate_le_wddm_budget": True}


def test_evaluate_alloc_gates_exact_chain():
    """F11 comparator semantics: measured_bound legs evaluate EXACTLY (no
    tolerance); a missing measurement can never pass (nest_gates F10)."""
    reserve = pf.ReservePolicy(retention_residual_bytes=0,
                               device_overhead_bytes=0,
                               external_margin_bytes=GIB)
    legs = pf.evaluate_alloc_gates(
        measured_used_bytes=10 * GIB, estimate_bytes=10 * GIB,
        measured_free_bytes=11 * GIB, reserve=reserve)
    assert legs == {"alloc_fits_wddm_budget": True,
                    "alloc_measured_le_estimate": True,
                    "alloc_estimate_le_wddm_budget": True}
    # One byte over the estimate is a FAILING GATE, not a note.
    legs = pf.evaluate_alloc_gates(
        measured_used_bytes=10 * GIB + 1, estimate_bytes=10 * GIB,
        measured_free_bytes=100 * GIB, reserve=reserve)
    assert legs["alloc_measured_le_estimate"] is False
    # estimate > budget fails independently (the leg the middle test
    # alone would miss: measured=20/estimate=40/budget=30).
    legs = pf.evaluate_alloc_gates(
        measured_used_bytes=20 * GIB, estimate_bytes=40 * GIB,
        measured_free_bytes=31 * GIB, reserve=reserve)
    assert legs["alloc_fits_wddm_budget"] is True
    assert legs["alloc_measured_le_estimate"] is True
    assert legs["alloc_estimate_le_wddm_budget"] is False
    # Missing measurements report None, never True.
    legs = pf.evaluate_alloc_gates(
        measured_used_bytes=None, estimate_bytes=GIB,
        measured_free_bytes=None, reserve=reserve)
    assert legs == {"alloc_fits_wddm_budget": None,
                    "alloc_measured_le_estimate": None,
                    "alloc_estimate_le_wddm_budget": None}


def test_gate_leg_names_match_the_n0_ledger():
    from gpuwm.verify import nest_gates

    ledger = {g.metric for g in nest_gates.gates_for("N0")}
    assert set(pf.N0_GATE_METRICS) == ledger
    for metric in pf.N0_GATE_METRICS:
        assert nest_gates.gate("N0", metric).kind == "measured_bound"


def test_recommend_column_chunk_lever(exp4, est4):
    # Comfortably large budget: the CONFIGURED chunk already fits, so the
    # lever recommends it unchanged.
    assert pf.recommend_column_chunk(
        exp4, est4.alloc_estimate_bytes) == exp4.column_chunk == 6250
    # Budget between the 3125 and 6250 estimates: halving lands on 3125 --
    # a host with less VRAM still gets walked back to the library default.
    e3125 = pf.estimate_experiment(exp4, column_chunk=3125)
    e6250 = pf.estimate_experiment(exp4, column_chunk=6250)
    budget = e3125.alloc_estimate_bytes + (
        e6250.alloc_estimate_bytes - e3125.alloc_estimate_bytes) // 2
    assert pf.recommend_column_chunk(exp4, budget) == 3125
    # No halving can fit a tiny budget.
    assert pf.recommend_column_chunk(exp4, GIB) is None


# ---------------------------------------------------------------------------
# CLI registrar (`gpuwm check` -- estimator mode is CPU-only)
# ---------------------------------------------------------------------------

def _run_check(argv):
    parser = argparse.ArgumentParser(prog="gpuwm")
    sub = parser.add_subparsers(dest="command", required=True)
    pf.register_cli(sub)
    args = parser.parse_args(argv)
    return args.func(args)


def test_check_cli_estimator_json(capsys):
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["experiment"] == "real74_4dom"
    assert payload["column_chunk"] == 6250
    # Ring-guard mp_ring_save_* saves add 23,815,680 B of per-domain
    # (arena-excluded) scratch across the four domains; x1.15 headroom
    # lands 27,388,032 B above the ports-branch CLI pin.
    assert payload["alloc_estimate_bytes"] == 22617453609
    # All requested moist-CQ slots are represented; the shared arena aliases
    # their lifetimes without changing the exact physical backing.
    assert payload["scratch_arena_saved_bytes"] == 6156502304
    assert payload["dycore_state_saved_bytes"] == 2869112700
    assert payload["domains"]["d04"]["by_category"]["nest"] == 193084928
    assert payload["gates"]["alloc_estimate_le_wddm_budget"] is True
    # Estimator-only mode: the measured legs stay unevaluated.
    assert payload["gates"]["alloc_measured_le_estimate"] is None
    assert "alloc" not in payload and "abort" not in payload
    # The reserve split proposal is reported for the controller.  The
    # overhead leg is no longer the 2026-07-16 zero-step probe constant
    # (1.39 GiB): it is this configuration's own measured non-pool
    # residency -- the CUDA context plus the local-memory backing store the
    # widest frame it LAUNCHES reserves.
    #
    # That frame used to be `kf_column`'s unspecialized 24,064 B, worth
    # 6,016,204,800 B of driver reservation.  `kf.cu`'s bound now compiles
    # to this case's own nz = 49, measured 9,216 B, which drops it BELOW
    # `ysu`'s 9,232 B -- so this configuration's widest launched frame is
    # now YSU's, and the reservation is 8,208 * 1536 * 170 = 2,143,272,960
    # B.  3,872.9 MiB of the old reserve was a compile-time array bound.
    # retention_residual is 3% of the alloc estimate, which carries the
    # ring lane's 27,388,032 B: +821,641 B over the ports-branch pin.
    assert payload["reserve_bytes"] == 3811652313
    assert payload["reserve_components"]["device_overhead_bytes"] == (
        pf.CUDA_CONTEXT_BYTES + 2143272960)
    assert payload["kernel_local_memory_bytes"] == 2143272960
    assert "kf" in payload["kernel_modules"]
    assert pf.kernel_local_frame_bytes(
        load_experiment_case(CONFIG_4DOM)[0])["kf"] == 9216
    assert payload["run_time_reserve_bytes"] == 6065468018
    assert payload["reserve_components"]["retention_residual_bytes"] == (
        math.ceil(0.03 * payload["alloc_estimate_bytes"]))


def test_check_cli_over_budget_fails_and_names_the_lever(capsys):
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "19.5"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "alloc_estimate_le_wddm_budget: FAIL" in out
    assert "OVER BUDGET" in out
    assert "--column-chunk 1562" in out


def test_check_over_budget_envelope_exits_nonzero(capsys, monkeypatch):
    """B-1: the report said "exceeds the WDDM budget" and exited 0.

    A node-7 pilot on virgin 1.0.1 read `gpuwm check`'s own sentence --
    "observed peak envelope 12.98 GiB exceeds the WDDM budget 11.64 GiB"
    -- out of a command that exited 0, so every script wrapping it read
    green.  The prose and the exit code cannot disagree; the prose is the
    accurate one.  4, not 1: no gate failed, and the levers differ.
    """
    monkeypatch.setattr(pf.sys, "platform", "win32")
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0, "a fitting envelope is still a clean pass"

    estimate_gib = payload["alloc_estimate_bytes"] / GIB
    tight = str(math.ceil(estimate_gib) + 1)
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", tight])
    out = capsys.readouterr().out
    assert "WARNING: observed peak envelope" in out
    assert rc == pf._EXIT_ENVELOPE_OVER_BUDGET
    assert rc != 0, "the sentence and the exit code must agree"
    # The warning names the code, so the reader of the text and the
    # reader of `echo $?` learn the same thing.
    assert f"exit code {pf._EXIT_ENVELOPE_OVER_BUDGET}" in out

    # A HARDER verdict outranks it: a failing gate is still 1, and a
    # fail-closed non-evaluable run is still 2.
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "19.5"])
    assert capsys.readouterr().out.count("OVER BUDGET") == 1
    assert rc == 1
    import sys
    monkeypatch.setitem(sys.modules, "cupy", None)
    rc = _run_check(["check", str(CONFIG_4DOM)])
    capsys.readouterr()
    assert rc == 2


def test_declared_free_is_capped_at_the_cards_physical_total(capsys):
    """B-2: `--card 16gb` declared 16.68 GiB free on a 16 GB card.

    The wizard states a budget and `check` adds the reserve back to
    recover a notional free.  That arithmetic never saw the card, so the
    16 GB tier bought the estimate about a gigabyte of budget the card
    does not physically have.  Free cannot exceed total, ever.
    """
    # The pure function first: declared size and measurement are both
    # ceilings, the tighter one binds, and neither ever widens.
    gib = int(GIB)
    assert pf.cap_free_to_physical(
        17 * gib, card_total_bytes=16 * gib,
        measured_total_bytes=None) == (16 * gib, 16 * gib)
    # A measurement of the same card is tighter than its nameplate size
    # (a "16 GB" card has ~15.57 GiB usable), and it wins.
    assert pf.cap_free_to_physical(
        17 * gib, card_total_bytes=16 * gib,
        measured_total_bytes=15 * gib) == (15 * gib, 15 * gib)
    # Already within capacity: untouched, and no cap is reported.
    assert pf.cap_free_to_physical(
        10 * gib, card_total_bytes=16 * gib,
        measured_total_bytes=15 * gib) == (10 * gib, None)
    # No capacity statement at all imposes no ceiling -- a ceiling that
    # cannot be measured must never be invented.
    assert pf.cap_free_to_physical(
        99 * gib, card_total_bytes=None,
        measured_total_bytes=None) == (99 * gib, None)

    # And end to end through the CLI, with the tier's own numbers: a
    # 16 GB card, the wizard's flat 3 GiB reserve, budget 13 GiB.  The
    # reserve check adds back is larger than 3 GiB, so the naive free
    # lands above the card.
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "13",
                     "--vram-gib", "16", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["reserve_bytes"] > 3 * gib, "otherwise nothing to cap"
    assert payload["measured_free_bytes"] <= 16 * gib
    assert payload["free_bytes_capped_to_physical_bytes"] is not None
    assert "capped" in payload["free_bytes_source"]
    # The budget follows the capped free, so the gate is evaluated
    # against VRAM that exists.
    assert payload["budget_bytes"] == (
        payload["measured_free_bytes"] - payload["reserve_bytes"])
    assert rc != 0, "22.6 GiB of estimate does not fit a 16 GB card"

    # Text mode says so out loud rather than only in --json.
    _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "13",
                "--vram-gib", "16"])
    out = capsys.readouterr().out
    assert "CAPPED" in out
    assert "free VRAM cannot exceed the card" in out

    # Without a card size the declared figure stands: --budget-gib is
    # how you size for a machine that is not this one.
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["free_bytes_capped_to_physical_bytes"] is None
    assert payload["measured_free_bytes"] > 100 * gib
    assert rc == 0


def test_check_cli_reports_observed_peak_envelope(capsys, monkeypatch):
    """The empirical envelope line: honest, informational, budget-aware.

    The Thompson rematch measured a machine peak of 1.746x the footprint
    projection (29,004 MiB vs 16.22 GiB); ``gpuwm check`` must surface
    footprint x1.75 as an OBSERVED envelope and warn when it exceeds the
    WDDM budget -- WITHOUT changing any gate, because the enforced
    numbers remain the itemized estimate and the measured legs.

    The exit code is NOT informational, though: see
    ``test_check_over_budget_envelope_exits_nonzero``.  A node-7 pilot
    read this command's rc 0 out of a report whose own text said the
    configuration might not fit.
    """
    monkeypatch.setattr(pf.sys, "platform", "win32")
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["observed_peak_envelope_platform"] == "windows"
    assert payload["observed_peak_envelope_factor"] == 1.75
    assert payload["observed_peak_envelope_bytes"] == int(
        payload["footprint_projection_bytes"] * 1.75)
    assert payload["observed_peak_envelope_bytes"] == (
        pf.observed_peak_envelope_bytes(
            payload["footprint_projection_bytes"]))
    # 100 GiB budget: envelope fits, no warning.
    assert payload["observed_peak_envelope_exceeds_budget"] is False
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OBSERVED PEAK ENVELOPE (x1.75 footprint, windows factor" in out
    assert "1.746x its footprint projection" in out
    assert "WARNING: observed peak envelope" not in out
    # The forecast is no longer the only phase this report prices, so the
    # historical line must say which phase it is, the preprocessing phase
    # must appear beside it, and one sentence must name the binding one.
    assert "FORECAST OBSERVED PEAK ENVELOPE" in out
    assert "INGEST OBSERVED PEAK ENVELOPE" in out
    assert "INGEST (preprocessing, --source era5)" in out
    assert "BINDING PHASE:" in out
    assert "memory-binding phase" in out

    # A budget the ESTIMATE fits but the envelope exceeds: the estimate
    # gate still passes, and the warning names the honest number.
    envelope_gib = payload["observed_peak_envelope_bytes"] / GIB
    estimate_gib = payload["alloc_estimate_bytes"] / GIB
    tight = str(math.ceil(estimate_gib) + 1)
    assert float(tight) < envelope_gib
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", tight,
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == pf._EXIT_ENVELOPE_OVER_BUDGET
    assert payload["gates"]["alloc_estimate_le_wddm_budget"] is True
    assert payload["observed_peak_envelope_exceeds_budget"] is True
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", tight])
    out = capsys.readouterr().out
    assert rc == pf._EXIT_ENVELOPE_OVER_BUDGET
    assert "WARNING: observed peak envelope" in out
    assert "exceeds the WDDM budget" in out
    # The warned number is the LARGEST phase, and the report says which.
    assert "BINDING PHASE:" in out
    assert payload["binding_phase"] in ("forecast", "ingest")
    assert payload["peak_envelope_bytes"] >= payload[
        "observed_peak_envelope_bytes"]
    # Estimator-only mode (no budget, no GPU): nothing to compare, no
    # false alarm.  cupy is stubbed out exactly as the fails-closed test
    # does so this leg never queries a real device.
    import sys
    monkeypatch.setitem(sys.modules, "cupy", None)
    rc = _run_check(["check", str(CONFIG_4DOM), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["budget_bytes"] is None
    assert payload["observed_peak_envelope_exceeds_budget"] is None


def test_peak_envelope_factor_is_platform_conditional():
    """WDDM is what 1.75 models, and Linux does not have it.

    Two instrumented Linux RTX 4090 pilots (2026-07-30) measured
    machine-wide peaks of 0.84x and 0.79x the footprint projection, so
    applying the Windows envelope there sold users roughly half the grid
    their cards could hold.
    """
    assert pf.PEAK_ENVELOPE_FACTORS == {"windows": 1.75,
                                        "windows-small": 1.45,
                                        "linux": 1.45}
    assert pf.OBSERVED_PEAK_OVER_FOOTPRINT == 1.75

    for name in ("win32", "cygwin", "msys"):
        assert pf.envelope_platform(name) == "windows"
        assert pf.peak_envelope_factor(name) == 1.75
    # WSL and Linux containers report `linux` too, which is the point.
    for name in ("linux", "linux2"):
        assert pf.envelope_platform(name) == "linux"
        assert pf.peak_envelope_factor(name) == 1.45

    footprint = 11_310_000_000
    assert pf.observed_peak_envelope_bytes(
        footprint, platform="win32") == int(footprint * 1.75)
    assert pf.observed_peak_envelope_bytes(
        footprint, platform="linux") == int(footprint * 1.45)


def test_an_unmeasured_platform_takes_the_conservative_accounting():
    """v1.0.0 gave every non-Windows name the Linux (optimistic) numbers.

    Only two platforms have measurements: Windows/WDDM (with Cygwin and
    MSYS, the same driver under another shell) and Linux (which is also
    what WSL and Linux containers report).  Everything else -- Darwin,
    a BSD, a name that does not exist yet -- was silently priced with
    the envelope that omits 4.12 GiB of fixed constants, on no evidence
    at all.  Fail-open is the wrong direction here: the Linux numbers
    are three runs on two Linux cards, not a default.
    """

    for name in ("darwin", "freebsd13", "sunos5", "emscripten"):
        assert not pf.platform_is_measured(name)
        assert pf.envelope_platform(name) == "windows"
        assert pf.peak_envelope_factor(name) == 1.75
        assert pf.platform_projection_constants(name) == (
            pf.pool_retention_residual_bytes(), pf.PROBE_DEVICE_OVERHEAD_BYTES)
        # ...and the substitution is announced, naming the platform.
        note = pf.unknown_platform_note(name)
        assert note is not None and name in note
        assert "no VRAM measurements" in note

        # The small-card experiment is not extended to it: that tier is
        # an experiment about WDDM, and an unmeasured platform is not
        # the place to run a second experiment on top of the first.
        assert pf.envelope_platform(name, vram_gib=8.0) == "windows"

    for name in ("win32", "cygwin", "msys", "linux", "linux2"):
        assert pf.platform_is_measured(name)
        assert pf.unknown_platform_note(name) is None


def test_the_projection_constants_are_platform_conditional_too():
    """The 1.75 multiplier was only half of it.

    ``pool_retention_residual_bytes`` (2.73 GiB) and
    ``PROBE_DEVICE_OVERHEAD_BYTES`` (1.39 GiB) are grid-independent
    Windows-pool constants.  At the wizard's smallest layout they are
    4.12 GiB of a 5.38 GiB projection -- 77% -- so no smaller grid could
    ever fit a 12 GiB card, whose GPU then sat 66% idle.  None of the
    three instrumented Linux runs showed them.
    """
    windows = pf.platform_projection_constants("win32")
    assert windows == (pf.pool_retention_residual_bytes(),
                       pf.PROBE_DEVICE_OVERHEAD_BYTES)
    assert sum(windows) / GIB == pytest.approx(4.12, abs=0.02)
    assert pf.platform_projection_constants("linux") == (0, 0)

    # On Linux the projection is the itemized alloc estimate, and the
    # envelope over it clears the worst measured peak/alloc ratio.
    for alloc_gib, measured_gib in ((7.20, 9.54), (7.29, 8.99),
                                    (3.51, 4.04)):
        envelope = pf.observed_peak_envelope_bytes(
            int(alloc_gib * GIB), platform="linux")
        assert measured_gib * GIB < envelope, (alloc_gib, measured_gib)


def test_check_cli_prints_the_linux_envelope_factor_when_on_linux(
        capsys, monkeypatch):
    """`gpuwm check` must say which platform factor it applied."""

    monkeypatch.setattr(pf.sys, "platform", "linux")
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["observed_peak_envelope_platform"] == "linux"
    assert payload["observed_peak_envelope_factor"] == 1.45
    assert payload["observed_peak_envelope_basis"] == (
        "measured-preliminary, 3 runs")
    assert payload["observed_peak_envelope_bytes"] == int(
        payload["footprint_projection_bytes"] * 1.45)
    # And the projection itself dropped the two Windows-pool constants.
    assert payload["footprint_projection_bytes"] == payload[
        "alloc_estimate_bytes"]
    assert payload["reserve_components"]["retention_residual_bytes"] >= 0

    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert ("OBSERVED PEAK ENVELOPE (x1.45 footprint, linux factor; "
            "measured-preliminary, 3 runs)") in out
    assert "1.15x to 1.32x the itemized alloc estimate" in out
    assert "1.746x its footprint projection" not in out
    assert "FORECAST OBSERVED PEAK ENVELOPE" in out
    assert "INGEST OBSERVED PEAK ENVELOPE" in out
    assert "BINDING PHASE:" in out


def test_check_cli_legacy_config_wraps(capsys):
    rc = _run_check(["check", str(CONFIG_D01), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert list(payload["domains"]) == ["d01"]
    # The 1024-descriptor health capacity adds 24,576 B and the ring-guard
    # saves 3,010,560 B to the pre-assembly pin (itemization-pin derivation).
    assert payload["domains"]["d01"]["resident_bytes"] == 1473642400


def test_check_cli_fails_closed_when_nothing_is_evaluable(capsys,
                                                          monkeypatch):
    """Fable F6 / shadow F5: estimator mode with no budget and no GPU
    verified NOTHING -- the exit code must say so (rc 2), never 0 via
    ``all([])``."""
    import sys
    monkeypatch.setitem(sys.modules, "cupy", None)  # import cupy fails
    rc = _run_check(["check", str(CONFIG_D01), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert all(v is None for v in payload["gates"].values())
    assert payload["budget_bytes"] is None


# ---------------------------------------------------------------------------
# Robust-5 failure paths (CPU, stubbed cupy -- no device touched)
# ---------------------------------------------------------------------------

class _StubPool:
    def __init__(self):
        self.freed = False

    def used_bytes(self):
        return 0

    def total_bytes(self):
        return 0

    def free_all_blocks(self):
        self.freed = True


def _stub_cupy(free_bytes, total_bytes=int(31.84 * GIB)):
    import types

    stub = types.ModuleType("cupy")

    class _OOM(Exception):
        pass

    stub.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(
            memGetInfo=lambda: (int(free_bytes), int(total_bytes)),
            deviceSynchronize=lambda: None),
        memory=types.SimpleNamespace(OutOfMemoryError=_OOM))
    pool = _StubPool()
    stub.get_default_memory_pool = lambda: pool
    return stub, pool


def test_headroom_abort_still_reports_and_exits_distinctly(capsys,
                                                           monkeypatch):
    """Fable F1 / shadow F5 fix: a headroom abort before measurement must
    still emit the structured report (estimate-side legs evaluated, abort
    reason recorded) with an exit code DISTINCT from a leg FAIL."""
    import sys
    stub, pool = _stub_cupy(free_bytes=2 * GIB)  # far short of the need
    monkeypatch.setitem(sys.modules, "cupy", stub)
    rc = _run_check(["check", str(CONFIG_D01), "--alloc", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 3  # aborted-before-measurement, not a leg FAIL (1)
    assert payload["abort"]["error"] == "PreflightHeadroomError"
    assert payload["abort"]["phase"] == "domain d01 construction"
    assert payload["abort"]["free_bytes"] == 2 * GIB
    # Estimate-side legs evaluated from the abort's measured free; the
    # measured legs stay None and can never pass.
    assert payload["gates"]["alloc_estimate_le_wddm_budget"] is False
    assert payload["gates"]["alloc_measured_le_estimate"] is None
    assert payload["gates"]["alloc_fits_wddm_budget"] is None
    assert payload["alloc_estimate_bytes"] > 0
    assert not pool.freed


def test_headroom_error_carries_structured_fields():
    stub, _ = _stub_cupy(free_bytes=1 * GIB)
    reserve = pf.ReservePolicy.flat(GIB // 2)
    with pytest.raises(pf.PreflightHeadroomError) as err:
        pf._require_headroom(stub, 2 * GIB, reserve, "unit fixture")
    exc = err.value
    assert exc.phase == "unit fixture"
    assert exc.free_bytes == GIB
    assert exc.reserve_bytes == GIB // 2
    assert exc.remaining_bytes == 2 * GIB


def test_alloc_oom_terminates_without_freeing(exp1, monkeypatch):
    """Robust-5 OOM policy: diagnostics + termination, NEVER
    ``free_all_blocks()``-and-continue.  Stubbed cupy + a DomainState
    that raises the stub's OutOfMemoryError -- no device involved."""
    import sys

    import gpuwm.core.state as state_mod

    stub, pool = _stub_cupy(free_bytes=64 * GIB, total_bytes=64 * GIB)
    monkeypatch.setitem(sys.modules, "cupy", stub)

    class _Boom:
        def __init__(self, cfg):
            raise stub.cuda.memory.OutOfMemoryError(
                "stub allocation failure")

    monkeypatch.setattr(state_mod, "DomainState", _Boom)
    with pytest.raises(pf.PreflightAllocError) as err:
        pf.run_alloc_preflight(exp1)
    assert err.value.phase == "domain d01 construction"
    assert "Terminating" in str(err.value)
    assert "column_chunk" in str(err.value)  # the first lever is named
    assert not pool.freed  # never free_all_blocks-and-continue


def test_alloc_preflight_materializes_and_injects_shared_workspaces(
        exp4, monkeypatch):
    """CPU/stubbed-CuPy proof that --alloc builds both shared workspaces."""
    import sys
    import types

    import gpuwm.core.state as state_mod
    import gpuwm.ingest.lateral_bc as lbc_mod

    # Keep the real four-domain geometry/registry but turn off physics so the
    # stub run has no unrelated CuPy allocation surface.
    domains = tuple(dataclasses.replace(
        dc, run=dataclasses.replace(
            dc.run, mp_physics=0, sf_sfclay_physics=0,
            sf_surface_physics=0, bl_pbl_physics=0,
            ra_physics=0, cu_physics=0)) for dc in exp4.domains)
    cpu_exp = dataclasses.replace(exp4, domains=domains)
    estimate = pf.estimate_experiment(cpu_exp)
    assert estimate.uses_shared_scratch_arena

    stub, pool = _stub_cupy(free_bytes=64 * GIB, total_bytes=64 * GIB)
    monkeypatch.setitem(sys.modules, "cupy", stub)
    scratch_sentinel = types.SimpleNamespace(
        nbytes=estimate.scratch_arena_bytes)
    dycore_sentinel = types.SimpleNamespace(
        nbytes=estimate.dycore_state_workspace_bytes)
    scratch_built = []
    dycore_built = []
    injected = []

    def build_scratch(domains_arg):
        scratch_built.append(tuple(domains_arg))
        return scratch_sentinel

    def build_dycore(domains_arg):
        dycore_built.append(tuple(domains_arg))
        return dycore_sentinel

    class _FakeState:
        def __init__(self, cfg, scratch_arena=None,
                     dycore_state_workspace=None):
            injected.append((scratch_arena, dycore_state_workspace))

        def scratch(self, shape, slot, dtype=None):
            return types.SimpleNamespace(shape=tuple(shape), nbytes=0)

    monkeypatch.setattr(
        state_mod, "build_shared_scratch_arena", build_scratch)
    monkeypatch.setattr(
        state_mod, "build_shared_dycore_state_workspace", build_dycore)
    monkeypatch.setattr(state_mod, "DomainState", _FakeState)
    monkeypatch.setattr(lbc_mod, "attach_lateral_boundaries",
                        lambda state, boundaries: None)

    report = pf.run_alloc_preflight(cpu_exp, reserve=pf.ReservePolicy.flat(0))
    assert scratch_built == [cpu_exp.domains]
    assert dycore_built == [cpu_exp.domains]
    assert injected == [
        (scratch_sentinel, dycore_sentinel)] * len(cpu_exp.domains)
    assert report.estimate.scratch_arena_bytes == scratch_sentinel.nbytes
    assert (report.estimate.dycore_state_workspace_bytes
            == dycore_sentinel.nbytes)
    assert pool.freed


# ---------------------------------------------------------------------------
# N0 allocation runs (controller-run GPU; enforced gates)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_alloc_preflight_d01_measured_le_estimate(exp1):
    report = pf.run_alloc_preflight(exp1)
    assert report.pool_used_peak_bytes > 0
    # The enforced estimator contract: measured > estimate is a FAILING
    # GATE, not a recalibration note.
    assert report.gates["alloc_measured_le_estimate"] is True
    # Zero steps, freed at exit: the pool must actually release.
    assert report.free_after_release_bytes > report.free_at_peak_bytes


@pytest.mark.gpu
def test_alloc_preflight_n0_four_domain():
    """N0 (gates all wave-2 ARC-B merges): the full manifest-driven
    allocation.  The budget legs are recorded for the controller's ledger
    adjudication; the estimator-correctness leg is asserted here.

    The probe is DOCUMENTED as a fresh-process tool, so the test runs it
    exactly as shipped -- a subprocess of the CLI -- and asserts on its
    JSON.  Before spawning, the PARENT must surrender its own device
    residue: earlier gpu tests leave DomainState<->PhysicsDriver reference
    CYCLES whose arrays survive free_all_blocks until a cycle collection
    (diagnosed 2026-07-16: 1.48 GiB retained pre-gc, 22 MB post-gc)."""
    import gc
    import json
    import subprocess
    import sys
    import cupy as cp
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    proc = subprocess.run(
        [sys.executable, "-m", "gpuwm.cli", "check",
         str(CONFIG_4DOM),
         "--alloc", "--reserve-gib", "1.888", "--json"],
        capture_output=True, text=True, cwd=ROOT, timeout=1200)
    # exit 1 = a budget leg failed (recorded for controller adjudication);
    # exit 2 = nothing evaluable is a genuine failure.  exit 3 = headroom
    # abort: under the full suite, session-scoped gpu fixtures hold device
    # memory no gc can surrender, so the fresh-process probe legitimately
    # cannot fit -- assert the abort JSON is well-formed, then skip (the
    # binding N0 evidence is the controller's standalone probe).  exit 4 =
    # every measured leg passed but the observed peak envelope sits above
    # the budget, which on a four-domain config is the expected verdict on
    # most cards; the measured legs below are what this test is about.
    assert proc.returncode in (0, 1, 3, 4), proc.stderr[-2000:]
    report = json.loads(proc.stdout)
    if proc.returncode == 3:
        assert report.get("abort"), "exit 3 must carry a structured abort"
        pytest.skip("N0 probe headroom-aborted under suite residency: "
                    + str(report["abort"])[:200])
    print("N0", report["gates"], "alloc", report["alloc"])
    assert report["gates"]["alloc_measured_le_estimate"] is True
    assert set(report["gates"]) == set(pf.N0_GATE_METRICS)
    assert report["alloc"]["pool_used_peak_bytes"] > 0


def test_legacy_rrtmg_variant_prices_the_call_peak_envelope():
    """Variant-aware preflight (assembly item): under the legacy 4/4
    variant the RRTMGP per-domain column shapes disappear (the legacy
    transients are the shared call-peak envelope instead), and the
    envelope itself is positive, chunk-bounded, and grows with ncol up
    to the chunk bound."""
    import dataclasses
    import pathlib

    from gpuwm.core.preflight import rrtmgp_column_shapes
    from gpuwm.core.rrtmg_legacy import legacy_radiation_vram_bytes
    from gpuwm.config import load_config

    legacy_cfg = load_config(
        pathlib.Path(__file__).resolve().parents[1]
        / "configs" / "real74_d01_rrtmg_legacy.toml")
    modern_cfg = dataclasses.replace(
        legacy_cfg, ra_rrtmg_variant="rte-rrtmgp",
        wrf_rrtmg_compatibility="none")
    assert rrtmgp_column_shapes(legacy_cfg, 10000.0) == {}
    assert rrtmgp_column_shapes(modern_cfg, 10000.0) != {}

    small = legacy_radiation_vram_bytes(
        ncol=1000, nz=legacy_cfg.nz, p_top=10000.0, column_chunk=None)
    big = legacy_radiation_vram_bytes(
        ncol=legacy_cfg.ny * legacy_cfg.nx, nz=legacy_cfg.nz,
        p_top=10000.0, column_chunk=None)
    assert 0 < small <= big
    # beyond the engine-default chunk bound the envelope must flatten
    # (chunking caps the transient, by construction of the engines)
    bigger = legacy_radiation_vram_bytes(
        ncol=4 * legacy_cfg.ny * legacy_cfg.nx, nz=legacy_cfg.nz,
        p_top=10000.0, column_chunk=None)
    assert bigger <= big * 2, (bigger, big)


def test_legacy_rrtmg_variant_prices_the_lw_chain_local_frame():
    """Variant-aware kernel-module selection (codex step-1 audit, major 1).

    ``ra_physics = 4`` is two implementations behind one selector value.
    Under ``ra_rrtmg_variant = 'rrtmg_legacy'`` the modern ``rrtmgp_*``
    kernels are never launched, and the legacy LW chain's widest kernel
    -- ``rlw_rtrn_march``, driver-measured 2,048 B/thread on sm_120
    (tests/test_rrtmg_lw_cuda.py ``LOCAL_FRAME_BOUNDS`` measurement
    record; docs/rrtmg_legacy_integration.md section 6) -- must be
    priced into the local-memory backing store instead of dropping out
    of the selector model fail-open.  The chained fragments themselves
    stay refused-without-measurement: they are covered by the measured
    composite translation units, never priced standalone.
    """
    legacy_cfg = load_config(
        ROOT / "configs" / "real74_d01_rrtmg_legacy.toml")
    start = datetime(1974, 4, 3, 12)
    exp = experiment_from_run_config(legacy_cfg, start)
    modules = pf.physics_kernel_modules(exp)
    # The legacy variant never launches the RTE+RRTMGP kernels ...
    assert not modules & {
        "rrtmgp_cloud", "rrtmgp_gas", "rrtmgp_mcica", "rrtmgp_rte"}
    # ... and does launch the device McICA twin plus the two chained TUs.
    assert {"rrtmg_mcica_wrf", "rrtmg_lw_legacy_chain",
            "rrtmg_sw_legacy"} <= modules

    frames = pf.kernel_local_frame_bytes(exp)
    assert frames["rrtmg_lw_legacy_chain"] == 2048
    assert frames["rrtmg_sw_legacy"] == 0  # post-spcvmc-restructure SW TU

    # Radiation alone (every masking selector stripped): the reservation
    # is exactly the LW chain's frame.  (2048 - 1024) x 1536 x 170 =
    # 267,386,880 B (~255 MiB) -- the incremental backing store the
    # fail-open selector omitted; the full ~510 MiB machine-wide store
    # at 2,048 B/thread includes the context's 1,024 B default-stack
    # half, which CUDA_CONTEXT_BYTES already carries.
    bare_run = dataclasses.replace(
        legacy_cfg, mp_physics=0, cu_physics=0, bl_pbl_physics=0,
        sf_sfclay_physics=0, sf_surface_physics=0)
    bare = experiment_from_run_config(bare_run, start)
    profile = pf.MEASURED_LOCAL_MEMORY_PROFILE
    assert (pf.kernel_local_memory_bytes(bare)
            == profile.reservation_bytes(2048) == 267386880)
    # The modern twin of the same bare selector set prices rrtmgp_rte's
    # 5,152 B module bound, as before this fix.
    modern_run = dataclasses.replace(
        bare_run, ra_rrtmg_variant="rte-rrtmgp",
        wrf_rrtmg_compatibility="none")
    assert (pf.kernel_local_memory_bytes(
                experiment_from_run_config(modern_run, start))
            == profile.reservation_bytes(5152))

    # Composite bookkeeping: the two measured TU frames cover exactly the
    # six legacy fragments that cannot compile standalone, so selecting a
    # fragment without a measurement still refuses (fail-closed), while a
    # legacy 4/4 request resolves to the composites and never trips it.
    covered = frozenset().union(
        *(tu.covers for tu in pf.CHAINED_TRANSLATION_UNIT_FRAMES.values()))
    assert covered == {
        "rrtmg_sw", "rrtmg_lw_chain", "rrtmg_lw_taugb02_10_11_12",
        "rrtmg_lw_taugb03_05", "rrtmg_lw_taugb06_09",
        "rrtmg_lw_taugb13_16"}
    assert covered <= pf.UNMEASURED_KERNEL_MODULES
    assert not modules & pf.UNMEASURED_KERNEL_MODULES

    # An unknown variant is refused, not priced from either row.
    unknown_run = dataclasses.replace(
        bare_run, ra_rrtmg_variant="rrtmg_v3")
    with pytest.raises(ValueError, match="ra_rrtmg_variant"):
        pf.physics_kernel_modules(
            experiment_from_run_config(unknown_run, start))


# ---------------------------------------------------------------------------
# The non-pool residency the CuPy pool never reports (measured 2026-07-26)
# ---------------------------------------------------------------------------
#
# Every number asserted below was MEASURED on the run host, either by
# bracketing a kernel's first launch with ``cudaMemGetInfo`` inside a real
# forecast or by sampling NVML device-wide at 1 s for a whole run.  Nothing
# here is derived from another estimate.

#: ``kf_column``'s first launch, measured twice in two separate traced
#: three-domain forecasts: 5,738.0 MiB of device memory that the pool did
#: not allocate and never gets back.
MEASURED_KF_RESERVATION_MIB = 5738.0

#: The same instrument on a synthetic kernel at five local-frame widths,
#: one block of 32 threads each: cumulative device growth over a bare
#: context, in MiB, keyed by per-thread local frame in bytes.
MEASURED_SYNTHETIC_RESERVATION_MIB = {
    1024: 2, 4096: 766, 8192: 1786, 16384: 3827, 24064: 5742}

CONFIG_4DOM_MYNN_KF = ROOT / "configs" / "real74_4dom_mynn_norad.toml"
CONFIG_4DOM_MYNN_NOCU = (
    ROOT / "configs" / "real74_4dom_mynn_norad_nocu.toml")


def test_the_local_memory_law_reproduces_every_measurement():
    """One allocation, sized by the widest launched frame, over the whole
    resident-thread capacity, minus the default stack the context already
    carries.  Six independent measurements, 1% tolerance."""
    profile = pf.MEASURED_LOCAL_MEMORY_PROFILE
    assert profile.resident_thread_capacity == 1536 * 170
    for local_bytes, measured_mib in (
            MEASURED_SYNTHETIC_RESERVATION_MIB.items()):
        predicted_mib = profile.reservation_bytes(local_bytes) / 1024 ** 2
        assert abs(predicted_mib - measured_mib) <= max(
            4.0, 0.01 * measured_mib), (
                f"{local_bytes} B/thread: model {predicted_mib:.1f} MiB "
                f"vs measured {measured_mib} MiB")
    kf_predicted = profile.reservation_bytes(
        pf.KERNEL_MAX_LOCAL_SIZE_BYTES["kf"]) / 1024 ** 2
    assert abs(kf_predicted - MEASURED_KF_RESERVATION_MIB) <= 1.0


def test_a_frame_inside_the_default_stack_reserves_nothing():
    """The 1024 B/thread baseline store belongs to the context, not to a
    kernel; MYNN's own kernels declare no static local frame at all."""
    profile = pf.MEASURED_LOCAL_MEMORY_PROFILE
    assert profile.reservation_bytes(1024) == 0
    assert profile.reservation_bytes(0) == 0
    assert pf.KERNEL_MAX_LOCAL_SIZE_BYTES["mynn_pbl"] == 0
    assert pf.KERNEL_MAX_LOCAL_SIZE_BYTES["mynn_surface"] == 0


def test_the_reservation_does_not_grow_with_domain_count():
    """The fingerprint that identified this term: it is a maximum over
    launched kernels, so three domains and four reserve the same bytes.

    The VALUE moved on 2026-07-26 when `kf.cu`'s KF_KMAX started compiling
    to the configuration's nz: 24,064 B/thread at the unspecialized 128,
    9,216 B at this case's 49 (both driver-measured).  The invariant this
    test exists for -- independence from domain count -- is unchanged.
    """
    exp4 = load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
    exp3 = dataclasses.replace(exp4, domains=exp4.domains[:3])
    assert {dc.run.nz for dc in exp4.domains} == {49}
    assert (pf.kernel_local_memory_bytes(exp3)
            == pf.kernel_local_memory_bytes(exp4)
            == pf.MEASURED_LOCAL_MEMORY_PROFILE.reservation_bytes(9216))
    # ... and it is still `kf` that sets it, now at the specialized width.
    assert pf.kernel_local_frame_bytes(exp4)["kf"] == 9216


def test_the_reservation_does_grow_with_the_level_count():
    """The term the specialization introduces, and the reason preflight now
    prices `kf`/`refl` per domain instead of from one module constant."""
    exp = load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
    deeper = dataclasses.replace(exp, domains=tuple(
        dataclasses.replace(dc, run=dataclasses.replace(dc.run, nz=98))
        for dc in exp.domains))
    # 188 B/level, rounded up to the frame's 8-byte granularity: 9,216 B at
    # 49 (9,212 padded) and 18,424 B at 98 (already aligned).
    assert pf.kernel_local_frame_bytes(exp)["kf"] == 9216
    assert pf.kernel_local_frame_bytes(deeper)["kf"] == 18424
    assert (pf.kernel_local_memory_bytes(deeper)
            > pf.kernel_local_memory_bytes(exp))
    # The unspecialized ceiling is still the ceiling: nothing may be priced
    # past the bound the source compiles to when nothing overrides it.
    over = dataclasses.replace(exp, domains=tuple(
        dataclasses.replace(dc, run=dataclasses.replace(dc.run, nz=129))
        for dc in exp.domains))
    with pytest.raises(ValueError, match="exceeds the KF_KMAX ceiling"):
        pf.kernel_local_memory_bytes(over)


def test_the_level_specialized_frame_model_agrees_with_every_measurement():
    """Six driver measurements on the RTX 5090 (2026-07-26), two bounds and
    three level counts per module.  Each is `align8(bytes_per_level * n)`."""
    measured = {
        ("kf", 128): 24064, ("kf", 49): 9216, ("kf", 30): 5640,
        ("refl", 256): 18432, ("refl", 49): 3528, ("refl", 30): 2160,
    }
    for (module, levels), frame in measured.items():
        spec = pf.LEVEL_SPECIALIZED_KERNEL_FRAMES[module]
        assert spec.frame_bytes(levels) == frame, (module, levels)
    # The unspecialized row of each is exactly the driver-measured module
    # maximum, so the two tables cannot drift apart unnoticed.
    for module, spec in pf.LEVEL_SPECIALIZED_KERNEL_FRAMES.items():
        assert (spec.frame_bytes(spec.unspecialized_levels)
                == pf.KERNEL_MAX_LOCAL_SIZE_BYTES[module])


def test_the_widest_frame_is_one_cumulus_kernel_not_anything_mynn_owns():
    exp = load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
    frames = pf.kernel_local_frame_bytes(exp)
    widest = max(frames, key=lambda m: frames[m])
    assert widest == "kf"
    without_cumulus = {m: f for m, f in frames.items() if m != "kf"}
    assert max(without_cumulus.values()) == 5120  # morrison
    # It is still the widest AS LAUNCHED, at a quarter of the frame it used
    # to carry: 9,216 B specialized to nz = 49 against 24,064 B at KF_KMAX
    # 128, and the runner-up unchanged.
    assert frames["kf"] == 9216
    assert pf.KERNEL_MAX_LOCAL_SIZE_BYTES["kf"] == 24064


def test_physics_kernel_modules_fails_closed_on_an_unpriced_selector():
    exp = load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
    d01 = exp.domains[0]
    bogus = dataclasses.replace(
        d01, run=dataclasses.replace(d01.run, mp_physics=55))
    broken = dataclasses.replace(exp, domains=(bogus,) + exp.domains[1:])
    with pytest.raises(ValueError,
                       match="no kernel-module row for mp_physics=55"):
        pf.physics_kernel_modules(broken)


def test_a_module_that_does_not_compile_is_refused_not_guessed():
    """Noah-MP's driver/energy/thermal kernels fail NVRTC at this checkout,
    so their local frame has never been measured.  A configuration that
    selects them must refuse, not price zero."""
    exp = load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
    noahmp = tuple(
        dataclasses.replace(dc, run=dataclasses.replace(
            dc.run, sf_surface_physics=4)) for dc in exp.domains)
    with pytest.raises(ValueError, match="do not compile at this checkout"):
        pf.kernel_local_memory_bytes(
            dataclasses.replace(exp, domains=noahmp))


def test_the_reflectivity_diagnostic_is_priced_only_when_it_can_fire():
    """``refl10cm_*`` is launched from the microphysics drivers'
    history-cadence branch alone.  The 60 s probes behind this model wrote
    their t=0 frames and never launched one."""
    exp = load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
    assert exp.run_seconds == 60.0
    assert not pf.refl_diagnostic_reachable(exp)
    assert "refl" not in pf.physics_kernel_modules(exp)
    production = dataclasses.replace(exp, run_seconds=43200.0)
    assert pf.refl_diagnostic_reachable(production)
    assert "refl" in pf.physics_kernel_modules(production)


def test_the_reflectivity_time_bomb_no_longer_moves_the_reservation():
    """FAILING FORM FIRST.

    The four-domain config WITHOUT cumulus is the one the reflectivity
    reservation could detonate: its widest launched frame before the first
    history frame is Morrison's 5,120 B (64 MiB), and
    `refl10cm_morrison_column` at the unspecialized REFL_KMAX = 256 carries
    18,432 B (4,335 MiB).  Neither traced probe ran long enough to launch
    it, so an as-built production forecast would have taken that step
    MID-FLIGHT, past the gate that let it start.  The first block
    reproduces the jump; the second shows it gone.
    """
    profile = pf.MEASURED_LOCAL_MEMORY_PROFILE
    probe = load_experiment_case(CONFIG_4DOM_MYNN_NOCU)[0]
    production = dataclasses.replace(probe, run_seconds=43200.0)
    assert "refl" not in pf.physics_kernel_modules(probe)
    assert "refl" in pf.physics_kernel_modules(production)

    # As LAUNCHED, from the driver: Morrison's sedimentation kernel measures
    # 1,280 B (64 MiB reserved -- the traced no-cumulus run's whole
    # local-memory term) and refl10cm_morrison_column 18,432 B.
    as_launched_jump = (profile.reservation_bytes(18432)
                        - profile.reservation_bytes(1280))
    assert round(as_launched_jump / 1024 ** 2) == 4271
    # As PRICED, with the module maximum preflight carries for Morrison --
    # over-priced by design, and still a 3,315 MiB mid-flight step.
    as_priced_jump = (profile.reservation_bytes(18432)
                      - profile.reservation_bytes(5120))
    assert round(as_priced_jump / 1024 ** 2) == 3315

    # Specialized to nz = 49 the same kernel measures 3,528 B against
    # Morrison's unchanged 5,120 B, so the widest launched frame does not
    # move at all when the first history frame comes due.
    assert pf.kernel_local_frame_bytes(production)["refl"] == 3528
    assert (pf.kernel_local_memory_bytes(production)
            == pf.kernel_local_memory_bytes(probe)
            == profile.reservation_bytes(5120))


def _rail_gate(config, *, rail_mib, other_mib, overhead_bytes=None):
    """The `gpuwm check` rail leg, with the card's occupancy supplied so the
    assertion does not depend on what the desktop happens to be holding."""
    exp = load_experiment_case(config)[0]
    estimate = pf.estimate_experiment(exp)
    reserve = pf.ReservePolicy.n0_alloc(
        exp, estimate_bytes=estimate.alloc_estimate_bytes)
    if overhead_bytes is not None:
        reserve = dataclasses.replace(
            reserve, device_overhead_bytes=overhead_bytes)
    free = pf.device_rail_free_bytes(
        rail_mib * 1024 ** 2, other_process_bytes=other_mib * 1024 ** 2)
    return pf.evaluate_alloc_gates(
        measured_used_bytes=None,
        estimate_bytes=estimate.alloc_estimate_bytes,
        measured_free_bytes=free, reserve=reserve)


def test_the_old_overhead_constant_passes_the_run_that_breached_the_rail():
    """FAILING FORM FIRST.

    ``configs/real74_4dom_mynn_norad.toml`` was RUN: 31,130 MiB device-wide
    against a 29,500 MiB rail, 1,630 MiB over, with 3,381 MiB of desktop on
    the card.  Preflight reported ``alloc_estimate_le_wddm_budget: PASS``
    beforehand.  This reproduces that pass with the 2026-07-16 zero-step
    probe overhead in place, so the fixed gate is never merely assumed to
    work.
    """
    legs = _rail_gate(CONFIG_4DOM_MYNN_KF, rail_mib=29500, other_mib=3381,
                      overhead_bytes=pf.PROBE_DEVICE_OVERHEAD_BYTES)
    assert legs["alloc_estimate_le_wddm_budget"] is True


def _as_built_overhead(config):
    """Non-pool overhead of the binary that RAN on 2026-07-26: CUDA context
    plus the reservation of the widest module frame at its UNSPECIALIZED
    bound, which is how ``kf``/``refl`` compiled before the bounds were
    specialized.  Keeps the historical measurements priced against the code
    that produced them."""
    exp = load_experiment_case(config)[0]
    widest = max(pf.KERNEL_MAX_LOCAL_SIZE_BYTES[m]
                 for m in pf.physics_kernel_modules(exp))
    return (pf.CUDA_CONTEXT_BYTES
            + pf.MEASURED_LOCAL_MEMORY_PROFILE.reservation_bytes(widest))


def test_the_measured_overhead_refuses_the_run_that_breached_the_rail():
    """The as-built binary, priced with the measured local-memory law: the
    run that went 1,630 MiB over is refused before it starts."""
    legs = _rail_gate(CONFIG_4DOM_MYNN_KF, rail_mib=29500, other_mib=3381,
                      overhead_bytes=_as_built_overhead(CONFIG_4DOM_MYNN_KF))
    assert legs["alloc_estimate_le_wddm_budget"] is False


def test_specializing_the_bound_is_what_lets_the_cumulus_config_through():
    """The same configuration, the same rail, the same desktop occupancy --
    the only thing that changed is that `kf.cu` compiles its column arrays
    to nz = 49 instead of 128.

    Measured, not projected.  `configs/real74_4dom_mynn_norad.toml` was RUN
    unchanged on 2026-07-26 after the specialization: 27,216 MiB device-wide
    peak (NVML, 1 Hz, 112 samples), status complete, 2,284 MiB under the
    29,500 MiB rail, against 31,130 MiB for the same file before.  See
    docs/kernel_local_memory_bounds.md.
    """
    legs = _rail_gate(CONFIG_4DOM_MYNN_KF, rail_mib=29500, other_mib=3381)
    assert legs["alloc_estimate_le_wddm_budget"] is True
    saved = (_as_built_overhead(CONFIG_4DOM_MYNN_KF)
             - pf.ReservePolicy.n0_alloc(
                 load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
             ).device_overhead_bytes)
    assert round(saved / 1024 ** 2) == 3698


def test_the_rail_gate_passes_the_four_domain_run_that_measured_under_it():
    """``configs/real74_4dom_mynn_norad_nocu.toml`` was RUN: 25,498 MiB
    device-wide, 4,002 MiB under the rail, same four domains, same
    861,001 columns."""
    legs = _rail_gate(CONFIG_4DOM_MYNN_NOCU, rail_mib=29500, other_mib=3464)
    assert legs["alloc_estimate_le_wddm_budget"] is True


def test_the_rail_never_widens_the_budget():
    """A rail is an ADDITIONAL ceiling.  A rail below what the card would
    hand out must bind; it can never hand out more."""
    exp = load_experiment_case(CONFIG_4DOM_MYNN_NOCU)[0]
    estimate = pf.estimate_experiment(exp)
    reserve = pf.ReservePolicy.n0_alloc(
        exp, estimate_bytes=estimate.alloc_estimate_bytes)
    rail_free = pf.device_rail_free_bytes(
        29500 * 1024 ** 2, other_process_bytes=3464 * 1024 ** 2)
    card_free = 40 * GIB
    assert min(card_free, rail_free) == rail_free
    assert reserve.budget_bytes(rail_free) < reserve.budget_bytes(card_free)


def test_the_non_pool_projection_brackets_both_measured_runs():
    """Estimate + non-pool residency, against the two device-wide peaks the
    runs actually reached (process share = device peak - desktop baseline).

    Both peaks were measured on 2026-07-26 by the AS-BUILT binary, before
    the ``kf``/``refl`` bounds were specialized, so the overhead leg is
    priced at the unspecialized frames those runs actually compiled to.
    """
    for config, other_mib, device_peak_mib in (
            (CONFIG_4DOM_MYNN_KF, 3381, 31130),
            (CONFIG_4DOM_MYNN_NOCU, 3464, 25498)):
        exp = load_experiment_case(config)[0]
        estimate = pf.estimate_experiment(exp)
        reserve = pf.ReservePolicy.n0_alloc(
            exp, estimate_bytes=estimate.alloc_estimate_bytes)
        projected = (estimate.alloc_estimate_bytes
                     + reserve.retention_residual_bytes
                     + _as_built_overhead(config))
        measured = (device_peak_mib - other_mib) * 1024 ** 2
        assert abs(projected - measured) / measured < 0.04, config.name
    # ... and the direction that mattered: the retired model under-projected
    # the run that breached, which is exactly how it passed it.
    exp = load_experiment_case(CONFIG_4DOM_MYNN_KF)[0]
    old = (pf.estimate_experiment(exp).alloc_estimate_bytes
           + pf.PROBE_DEVICE_OVERHEAD_BYTES)
    assert old < (31130 - 3381) * 1024 ** 2


@pytest.mark.gpu
def test_the_recorded_local_frames_match_the_driver():
    """Regenerate ``KERNEL_MAX_LOCAL_SIZE_BYTES`` from NVRTC + the driver.

    A kernel that grows its per-thread frame silently moves the whole
    process's device footprint by gigabytes, so the table is not allowed to
    go stale.

    The default-stack-limit leg is measured in a FRESH SUBPROCESS, and that
    is what it proves: ``cudaLimitStackSize`` is process state, not a device
    constant.  The CUDA runtime raises it for the life of the process the
    first time a fatter-framed kernel is loaded (kf/refl at 24,064 B against
    the 1,024 B fresh default -- nothing in this tree calls
    ``deviceSetLimit``), so reading it in the suite process measures which
    tests happened to run first, which made this test pass alone and fail
    after the kf/refl modules in a one-process full run.  The reservation
    law prices what a FRESH gpuwm process reserves, so the fresh-process
    value is the only one the recorded constant may be compared against;
    the assertion is exact, the same production reader runs in the probe,
    and no launch order in this process can move the answer.
    """
    import re
    import subprocess
    import sys

    cp = pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module

    kdir = ROOT / "gpuwm" / "core" / "kernels"
    symbol = re.compile(
        r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)')
    probe = subprocess.run(
        [sys.executable, "-c",
         "import cupy as cp\n"
         "from gpuwm.core import preflight as pf\n"
         "print(pf.local_memory_profile_from_device(cp)"
         ".default_stack_limit_bytes)"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    assert probe.returncode == 0, probe.stderr[-2000:]
    assert int(probe.stdout.strip().splitlines()[-1]) == (
        pf.MEASURED_LOCAL_MEMORY_PROFILE.default_stack_limit_bytes)

    observed = {}
    uncompilable = set()
    for path in sorted(kdir.glob("*.cu")):
        try:
            module = load_module(path.stem)
        except Exception:  # noqa: BLE001  -- recorded here, repaired elsewhere
            uncompilable.add(path.stem)
            continue
        widest = 0
        for name in sorted(set(symbol.findall(path.read_text()))):
            try:
                attributes = module.get_function(name).attributes
            except Exception:  # noqa: BLE001
                continue
            widest = max(widest, int(attributes["local_size_bytes"]))
        observed[path.stem] = widest
    assert uncompilable == set(pf.UNMEASURED_KERNEL_MODULES)
    assert observed == pf.KERNEL_MAX_LOCAL_SIZE_BYTES

# ---------------------------------------------------------------------------
# Noah-MP land-surface transients (noahmp_lsm_transient_shapes)
# ---------------------------------------------------------------------------
# These live here, not in tests/test_noahmp_column_slab.py, because that
# module's helpers import cupy and conftest._cupy_scope therefore marks the
# whole file gpu -- and a pricing gate that only runs where a device happens
# to be present is the exact blindness the allocation ratchet was moved out
# of test_mynn_pbl_scratch.py to escape.  Nothing below opens a device.

@pytest.mark.parametrize("land_surface,nsoil", [(3, 9), (4, 4)])
def test_mynn_lsm_pairings_price_the_union_of_both_components(
        land_surface, nsoil):
    """The newly reachable tuples cannot shed either side's persistent state."""
    from gpuwm.core.mynn_pbl_runtime import MYNN_PBL_STATE_3D
    from gpuwm.core.mynn_sfclay import MYNN_SURFACE_OUTPUTS

    cfg = RunConfig(
        nx=8, ny=6, nz=12, dx=1000.0, dy=1000.0, ztop=12000.0,
        dt=5.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=6, sf_sfclay_physics=5, bl_pbl_physics=5,
        sf_surface_physics=land_surface, num_soil_layers=nsoil)
    shapes = pf.physics_array_shapes(cfg)
    for name in MYNN_SURFACE_OUTPUTS:
        assert shapes[f"fields/{name}"] == (cfg.ny, cfg.nx)
    for name in MYNN_PBL_STATE_3D:
        assert shapes[f"fields/{name}"] == (cfg.nz, cfg.ny, cfg.nx)

    if land_surface == 3:
        from gpuwm.core.ruc_runtime import RUC_STATE_2D, RUC_STATE_3D
        for name in MYNN_SURFACE_OUTPUTS:
            assert shapes[f"fields/{name}_sea"] == (cfg.ny, cfg.nx)
        for name in RUC_STATE_2D:
            assert shapes[f"fields/{name}"] == (cfg.ny, cfg.nx)
        for name in RUC_STATE_3D:
            assert shapes[f"fields/{name}"] == (nsoil, cfg.ny, cfg.nx)
    else:
        from gpuwm.core.noahmp_runtime import (
            NOAHMP_STATE_2D, NOAHMP_STATE_SNOWSOIL_3D,
            NOAHMP_STATE_SNOW_3D, NSNOW,
        )
        for name in NOAHMP_STATE_2D:
            assert shapes[f"fields/{name}"] == (cfg.ny, cfg.nx)
        for name in NOAHMP_STATE_SNOW_3D:
            assert shapes[f"fields/{name}"] == (NSNOW, cfg.ny, cfg.nx)
        for name in NOAHMP_STATE_SNOWSOIL_3D:
            assert shapes[f"fields/{name}"] == (
                NSNOW + nsoil, cfg.ny, cfg.nx)


def test_preflight_prices_the_bound_the_runtime_launches_with():
    """The transient term reads the runtime's own constants, by name.

    ``SLAB_COLUMN_CHUNK`` is the explicit column-chunk bound the slab
    modules' allocation-inventory rows demanded; this is the assertion that
    the number preflight prices IS that bound and not a copy that can drift.
    """
    from gpuwm.config import RunConfig
    from gpuwm.core.noahmp_runtime import (
        COLUMN_BATCH, SLAB_COLUMN_CHUNK, SLAB_GRID_TRANSIENT_BYTES_PER_COLUMN,
        SLAB_TRANSIENT_BYTES_PER_COLUMN)
    from gpuwm.core.preflight import noahmp_lsm_transient_shapes

    base = dict(nx=600, ny=600, nz=40, dx=1000.0 / 3.0, dy=1000.0 / 3.0,
                ztop=16000.0, dt=5.0 / 3.0, run_seconds=0.0,
                time_step_sound=4, moist=True, mp_physics=6,
                sf_sfclay_physics=1, bl_pbl_physics=1, bldt=0.0)
    shapes = noahmp_lsm_transient_shapes(
        RunConfig(sf_surface_physics=4, **base))
    assert shapes["noahmp_lsm/slab_chunk_transients"] == (
        min(SLAB_COLUMN_CHUNK, 360000), SLAB_TRANSIENT_BYTES_PER_COLUMN)
    assert shapes["noahmp_lsm/slab_grid_transients"] == (
        360000, SLAB_GRID_TRANSIENT_BYTES_PER_COLUMN)
    assert shapes["noahmp_lsm/staged_leaf_batches"] == (
        min(COLUMN_BATCH, 360000), 620)
    # A domain smaller than the chunk prices its own width, not the bound.
    small = noahmp_lsm_transient_shapes(
        RunConfig(sf_surface_physics=4, **{**base, "nx": 8, "ny": 6}))
    assert small["noahmp_lsm/slab_chunk_transients"] == (
        48, SLAB_TRANSIENT_BYTES_PER_COLUMN)
    # And a run without Noah-MP pays nothing.
    assert noahmp_lsm_transient_shapes(
        RunConfig(sf_surface_physics=2, **base)) == {}


def test_estimate_domain_carries_the_noahmp_transient_items():
    """The term is in the estimate a launcher actually reads, not only in a
    helper a launcher could forget to call."""
    from gpuwm.config import RunConfig
    from gpuwm.core.preflight import estimate_domain
    from gpuwm.experiment import DomainConfig

    cfg = RunConfig(nx=64, ny=64, nz=40, dx=1000.0 / 3.0, dy=1000.0 / 3.0,
                    ztop=16000.0, dt=5.0 / 3.0, run_seconds=0.0,
                    time_step_sound=4, moist=True, mp_physics=6,
                    sf_sfclay_physics=1, sf_surface_physics=4,
                    bl_pbl_physics=1, bldt=0.0)
    estimate = estimate_domain(DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=3600.0, run=cfg, time_step=1))
    names = {item.name: item for item in estimate.items
             if item.name.startswith("noahmp_lsm/")}
    assert set(names) == {"noahmp_lsm/slab_chunk_transients",
                          "noahmp_lsm/slab_grid_transients",
                          "noahmp_lsm/staged_leaf_batches"}
    for item in names.values():
        assert item.category == "transient"
        assert item.itemsize == 1


# ---------------------------------------------------------------------------
# Preprocessing (ingest) phase -- the phase nobody used to price
# ---------------------------------------------------------------------------


def test_ingest_analysis_is_sized_by_source_levels_and_target_grid(exp1):
    """Horizontal interpolation lands source LEVELS on the MODEL grid."""
    run = exp1.root.run
    shapes = pf.ingest_analysis_shapes(run, source="gfs")
    levels = pf.SOURCE_ANALYSIS_LEVELS["gfs"]
    mass = [s for n, s in shapes.items() if n.startswith("analysis_mass")]
    assert mass and all(s == (levels, run.ny, run.nx) for s in mass)
    assert shapes["analysis_u_level_0"] == (levels, run.ny, run.nx + 1)
    assert shapes["analysis_v_level_0"] == (levels, run.ny + 1, run.nx)
    surface = [s for n, s in shapes.items() if n.startswith("analysis_surf")]
    assert len(surface) == pf.SOURCE_ANALYSIS_SURFACE_FIELDS["gfs"]
    assert all(s == (run.ny, run.nx) for s in surface)
    # ERA5 carries more levels, so its analysis is strictly larger.
    era5 = pf.ingest_analysis_shapes(run, source="era5")
    assert (math.prod(era5["analysis_mass_level_0"])
            > math.prod(shapes["analysis_mass_level_0"]))
    with pytest.raises(ValueError, match="no forcing-analysis level"):
        pf.ingest_analysis_shapes(run, source="not-a-product")


def test_ingest_state_term_is_the_real_domain_state_inventory(exp1):
    """Ingest builds a full DomainState per time -- the same one priced
    for the forecast, not a smaller setup-only object."""
    est = pf.estimate_ingest(exp1, source="gfs")
    expected = sum(4 * math.prod(shape) for shape
                   in pf.state_array_shapes(exp1.root.run).values())
    assert est.category_bytes("state") == expected


def test_ingest_holds_two_forcing_times_not_all_of_them(exp1):
    """The defect in one assertion.

    Ingest used to keep every forcing time's analysis AND state resident,
    which is why preprocessing a 24 h GFS case peaked at roughly twice the
    forecast.  Streaming keeps two.  Both numbers are reported so a user
    can see what the phase would have cost.
    """
    est = pf.estimate_ingest(
        exp1, source="gfs", forcing_interval_seconds=10800.0)
    assert est.resident_times == pf.INGEST_RESIDENT_FORCING_TIMES == 2
    assert est.n_forcing_times > est.resident_times
    assert (est.resident_bytes
            == 2 * est.per_time_bytes + est.forcing_table_bytes)
    assert (est.unstreamed_resident_bytes
            == est.n_forcing_times * est.per_time_bytes
            + est.forcing_table_bytes)
    # Every time beyond the two resident ones is pure saving.
    assert (est.unstreamed_resident_bytes - est.resident_bytes
            == (est.n_forcing_times - est.resident_times)
            * est.per_time_bytes)
    # The retained perimeter frames are what replaced those states, and
    # all of them together stay far below one of them.
    assert est.boundary_frame_bytes < est.per_time_bytes


def test_ingest_envelope_is_conservative_against_the_measured_case():
    """Both ends of the CONUS 12 km measurement, re-derived here.

    Measured on an RTX 5090 (process-attributed peak, 432 MiB CUDA
    context included): 15,288 MiB before the streaming fix and 4,672 MiB
    after, same case, byte-identical outputs.  The estimate must bound
    BOTH -- a sizing number that lands under a measured peak is the
    failure mode this whole phase estimate exists to prevent.
    """
    run = RunConfig(nx=414, ny=330, nz=49, dx=12000.0, dy=12000.0,
                    ztop=20000.0, dt=60.0, run_seconds=86400.0,
                    mp_physics=10, moist=True, terrain_opt=1,
                    spec_bdy_width=5, specified=True)
    exp = experiment_from_run_config(run, datetime(2026, 7, 30))
    est = dataclasses.replace(
        pf.estimate_ingest(exp, source="gfs",
                           forcing_interval_seconds=10800.0),
        device_overhead_bytes=0)  # the measured node is Linux
    assert est.n_forcing_times == 9

    measured_after = 4672 * 1024 ** 2
    assert est.peak_envelope_bytes >= measured_after
    assert est.peak_envelope_bytes <= 1.30 * measured_after

    measured_before = 15288 * 1024 ** 2
    before = (math.ceil(est.headroom * (est.unstreamed_resident_bytes
                                        + est.transient_bytes))
              + est.context_bytes)
    assert before >= measured_before
    assert before <= 1.30 * measured_before


def test_phase_estimate_names_the_binding_phase_and_the_number(exp1):
    phases = pf.estimate_phases(exp1, source="gfs")
    assert phases.binding_phase in ("forecast", "ingest")
    assert phases.peak_envelope_bytes == max(
        phases.forecast_envelope_bytes, phases.ingest_envelope_bytes)
    budget = phases.peak_envelope_bytes - 1
    assert not phases.fits(budget)
    verdict = phases.verdict(budget)
    assert "EXCEEDS" in verdict
    assert f"{phases.peak_envelope_bytes / GIB:.2f}" in verdict
    assert "forecast" in verdict and "ingest" in verdict
    assert phases.fits(phases.peak_envelope_bytes)
    assert "fits" in phases.verdict(phases.peak_envelope_bytes)


def test_config_forcing_source_refuses_to_guess(tmp_path):
    """A config whose source cannot be priced says so; it never lets the
    forecast number stand in for the whole run."""
    # A plain RunConfig TOML is not an experiment TOML at all, so it
    # records no forcing product and must not be guessed at.
    plain = tmp_path / "plain.toml"
    plain.write_text(CONFIG_D01.read_text(encoding="utf-8"),
                     encoding="utf-8")
    assert pf.config_forcing_source(plain) is None
    note = pf.unpriced_ingest_note(plain)
    assert "NOT PRICED" in note and "FORECAST only" in note
    assert "no forcing product at all" in note
    named = pf.unpriced_ingest_note(plain, "hrrr")
    assert "--source hrrr" in named and "does not model" in named


def test_an_unpriced_source_is_said_out_loud_not_scored_zero(exp1):
    """HRRR's ingest is a different lane and nothing here measured it.

    Returning zero for a phase you did not model reads exactly like
    "this phase is free", which is the failure this estimate exists to
    end.  So it is reported absent, the forecast stands alone, and the
    verdict says which of the two happened.
    """
    priced = pf.estimate_phases(exp1, source="gfs")
    unpriced = pf.estimate_phases(exp1, source="hrrr")
    assert priced.ingest_priced and priced.ingest is not None
    assert not unpriced.ingest_priced
    assert unpriced.ingest is None
    assert unpriced.ingest_envelope_bytes is None
    assert unpriced.binding_phase == "forecast"
    assert (unpriced.peak_envelope_bytes
            == unpriced.forecast_envelope_bytes)
    assert "NOT PRICED" in unpriced.verdict(None)
    assert "hrrr" in unpriced.verdict(None)
    assert pf.estimate_phases(exp1, source=None).ingest is None
