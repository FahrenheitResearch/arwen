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
    both fails here (full-featured config exercises every conditional).

    The union must span every scheme that allocates something no other
    scheme does, because it is asserted as an EQUALITY in both directions.
    ``mp_physics=18`` covers the NSSL scalars; ``mp_physics=28`` covers
    Thompson aerosol-aware's nwfa/nifa/nc0/nwfa0/nifa0 and the two 2-D
    surface emission fields.  Widening this union is the sanctioned way to
    admit a new scheme; TRIMMING the classification sets is not -- an
    attribute dropped from the restart manifest is a field that silently
    vanishes across a checkpoint.
    """
    from gpuwm.config import SASE_PBL_SCHEME

    nssl_cfg = dataclasses.replace(d01_cfg, mp_physics=18)
    # km_opt=2 is the only configuration that allocates the prognostic-TKE
    # carrier, so the union needs an LES-closure arm or tke/tke0 would be
    # classified in the restart manifest and unaccounted for in the VRAM
    # projection (validate_run_config is deliberately not run here -- this
    # is a shape manifest, not an admissible run).
    les_cfg = dataclasses.replace(
        d01_cfg, km_opt=2, bl_pbl_physics=0, khdif=0.0, kvdif=0.0)
    aerosol_cfg = dataclasses.replace(d01_cfg, mp_physics=28)
    # The SASE closure owns one conditional prognostic that no other
    # configuration allocates, so it joins the union for the same reason
    # NSSL does: this test's whole claim is that every conditional is
    # exercised.
    sase_cfg = dataclasses.replace(d01_cfg,
                                   bl_pbl_physics=SASE_PBL_SCHEME,
                                   km_opt=0, khdif=0.0, kvdif=0.0,
                                   bldt=0.0)
    names = (set(pf.state_array_shapes(d01_cfg))
             | set(pf.state_array_shapes(nssl_cfg))
             | set(pf.state_array_shapes(les_cfg))
             | set(pf.state_array_shapes(aerosol_cfg))
             | set(pf.state_array_shapes(sase_cfg)))
    classified = (set(restart.STATE_SERIALIZED_ATTRS)
                  | set(restart.STATE_REBUILT_ATTRS)
                  | set(restart.STATE_SETUP_ARRAYS))
    assert names == classified
    # And the specific names this port added, so a later edit cannot make
    # the equality hold again by deleting them from BOTH sides.
    for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
        assert name in restart.STATE_SERIALIZED_ATTRS
    for name in ("nc0", "nwfa0", "nifa0"):
        assert name in restart.STATE_REBUILT_ATTRS


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
    """The UP_HELI_MAX lane costs FIVE (ny, nx) FP32 planes and nothing
    else; the flagship (nwp_diagnostics = 0) registry is untouched.

    Three were the diagnostic's own (the accumulator plus two per-launch
    work planes).  The other two are the consumer-owned tracking windows
    added 2026-08-07: same running-max operator, folded in the same pass,
    but reset by the consumer that reads them instead of by the history
    writer, so a storm-following nest's placement stopped depending on
    the output cadence.  They are priced on this gate because that is the
    gate that allocates them.
    """
    base = pf.scratch_slot_registry(d01_cfg, n_lbc_intervals=2)
    on_cfg = dataclasses.replace(d01_cfg, nwp_diagnostics=1)
    on = pf.scratch_slot_registry(on_cfg, n_lbc_intervals=2)
    added = {"up_heli_max", "uh_diag_col", "uh_diag_use",
             "uh_follow_window", "uh_spawn_window"}
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
        # Every microphysics scheme with slots of its own, so a new family
        # cannot land unaudited.  km_opt=4 turns on the smag_r* held
        # tendencies, which is where the mp=28 number/aerosol rows live.
        RunConfig(**_TINY, moist=True, mp_physics=8, km_opt=4),
        RunConfig(**_TINY, moist=True, mp_physics=18, km_opt=4),
        RunConfig(**_TINY, moist=True, mp_physics=28, km_opt=4),
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
                RunConfig(**_TINY, moist=True, mp_physics=28),
                RunConfig(**_TINY, sf_sfclay_physics=1),
                RunConfig(**_TINY, nwp_diagnostics=1),
                # LES closures: km_opt=3 owns the vertical exchange-
                # coefficient pair, km_opt=2 adds the prognostic-TKE
                # carrying slots and (with the toggle on) the report-only
                # budget family.  Without these arms the whole LES slot
                # family is invisible to this completeness gate.
                RunConfig(**_TINY, km_opt=3, bl_pbl_physics=0),
                RunConfig(**_TINY, km_opt=2, bl_pbl_physics=0,
                          tke_budget=1)):
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


# ---------------------------------------------------------------------------
# mp_physics=28 -- Thompson aerosol-aware.  These pin the state, scratch,
# nest and pricing inventories the rest of the port builds on, and they pin
# the two places a mistake would be invisible: the transport discriminator
# and the mp=8/mp=10 non-interference.
# ---------------------------------------------------------------------------

_MP28 = dict(moist=True, moist_cq=True, mp_physics=28)


def test_mp28_state_allocation_inventory():
    cfg = RunConfig(**_TINY, **_MP28)
    shapes = pf.state_array_shapes(cfg)
    mass = (cfg.nz, cfg.ny, cfg.nx)
    surface = (cfg.ny, cfg.nx)
    # Prognostic scalars + their RK time-t copies.
    for name in ("nc", "nr", "ni", "nwfa", "nifa",
                 "nc0", "nr0", "ni0", "nwfa0", "nifa0",
                 "qi", "qs", "qg", "qi0", "qs0", "qg0",
                 "effc", "effi", "effs"):
        assert shapes[name] == mass, name
    # Surface emission tendencies, 2-D and cross-step constant.
    assert shapes["nwfa2d"] == surface
    assert shapes["nifa2d"] == surface
    # Thompson has no effr; Morrison/NSSL-only fields must not appear.
    for name in ("effr", "ns", "ng", "ns0", "ng0", "qh", "qnn"):
        assert name not in shapes, name


def test_mp28_state_allocation_matches_the_real_domain_state():
    """The shape manifest is a transcription of state.py; prove it.

    A manifest that drifts from the constructor is worse than no manifest:
    the arena is sized from the manifest and bound to the constructor.
    """
    import types

    import gpuwm.core.state as state_mod

    cfg = RunConfig(**_TINY, **_MP28)
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(state_mod, "cp", np)
        state = state_mod.DomainState(cfg)
    finally:
        monkey.undo()
    declared = pf.state_array_shapes(cfg)
    actual = {name: tuple(value.shape)
              for name, value in vars(state).items()
              if isinstance(value, np.ndarray)}
    assert actual == declared
    assert isinstance(state, types.SimpleNamespace) is False  # sanity


def test_mp28_transports_droplet_and_aerosol_number_but_mp10_does_not():
    """The transport gate.  This is the one mp=28 decision whose mistake is
    both silent and expensive: mp_physics=10 ALREADY allocates ``state.nc``
    and deliberately does not transport it, so a presence-of-``nc`` test
    would start advecting Morrison's diagnostic droplet number through every
    generic dycore consumer and move a validated trajectory.  The
    discriminator must be ``nwfa``, which exactly one scheme allocates.
    """
    import gpuwm.core.state as state_mod
    from gpuwm.core import moist

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(state_mod, "cp", np)
        mp8 = state_mod.DomainState(
            RunConfig(**_TINY, moist=True, mp_physics=8))
        mp10 = state_mod.DomainState(
            RunConfig(**_TINY, moist=True, mp_physics=10))
        mp28 = state_mod.DomainState(RunConfig(**_TINY, **_MP28))
    finally:
        monkey.undo()

    # The frozen receipts (tests/test_mp8_frozen.py R3) restated here so a
    # change to moist.py fails in its own test file too.
    assert moist.extra_moist_species(mp8) == ("qi", "qs", "qg", "nr", "ni")
    assert moist.extra_moist_species(mp10) == (
        "qi", "qs", "qg", "nr", "ni", "ns", "ng")
    assert moist.extra_moist_species(mp28) == (
        "qi", "qs", "qg", "nr", "ni", "nc", "nwfa", "nifa")

    # Morrison really does own an nc that is really not transported.
    assert getattr(mp10, "nc", None) is not None
    assert "nc" not in moist.extra_moist_species(mp10)
    assert getattr(mp10, "nc0", None) is None
    # ... and the discriminator is unique to mp=28.
    assert getattr(mp8, "nwfa", None) is None
    assert getattr(mp10, "nwfa", None) is None
    assert getattr(mp28, "nwfa", None) is not None
    # The generic filter itself must not have been widened.
    assert moist.TRANSPORTED_NUMBER_SPECIES == ("nr", "ni", "ns", "ng")


def test_mp28_scratch_registry_is_complete():
    cfg = RunConfig(**_TINY, **_MP28)
    slots = pf.scratch_slot_registry(cfg)
    mass = (cfg.nz, cfg.ny, cfg.nx)
    surface = (cfg.ny, cfg.nx)
    classic = {
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
    }
    assert classic.items() <= slots.items()
    aerosol = {name: mass for name in (
        "mp_thompson_aero_ncten",
        "mp_thompson_aero_nwfaten",
        "mp_thompson_aero_nifaten",
        "mp_thompson_aero_entry_density",
        "mp_thompson_aero_nwfa_entry_m3",
        "mp_thompson_aero_nifa_entry_m3",
        "mp_thompson_aero_tau1_density",
        "mp_thompson_aero_nwfa_work_m3",
        "mp_thompson_aero_qc_entry",
        "mp_thompson_aero_ni_entry",
        "mp_thompson_aero_rc_entry",
        "mp_thompson_aero_nc_entry_m3",
        "mp_thompson_aero_nu_c_entry",
        "mp_thompson_aero_l_qc_entry",
        "mp_thompson_aero_condensation_rate",
    )}
    assert aerosol.items() <= slots.items()
    # All fifteen are arena-eligible, and the audit row that says so is a
    # write_before_read row with real evidence.
    for slot in aerosol:
        assert pf.scratch_slot_uses_arena(slot), slot
        row = pf.scratch_slot_lifetime(slot)
        assert row is not None and row.kind == "write_before_read"
        assert row.evidence and row.rationale
    # mp=28 owns qi/qs, so the physics prep must NOT substitute zero planes.
    assert "physics_qi" not in slots and "physics_qs" not in slots
    # The aerosol slots belong to mp=28 alone.
    mp8_slots = set(pf.scratch_slot_registry(
        RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=8)))
    assert not (set(aerosol) & mp8_slots)


def test_mp28_every_scratch_slot_is_classified_for_restart():
    """gpuwm/io/restart.py.  ``classify_scratch_slot`` fails CLOSED, and its
    ``mp_`` rule is exact-names-only precisely so a new accumulator cannot be
    silently dropped from a checkpoint.  Every slot mp=28 can create must
    therefore carry an explicit classification.

    All fifteen aerosol slots are ``rebuild``, and that is the physics: WRF
    zeroes ncten/nwfaten/nifaten at the top of every column call
    (module_mp_thompson.F:1679-1681) and applies them once before returning
    (:3972-4021).  Serializing a tendency that has already been applied would
    apply it a second time on the resumed step.
    """
    from gpuwm.io import restart as restart_mod

    slots = set(pf.scratch_slot_registry(
        RunConfig(**_TINY, **_MP28, km_opt=4, diff_6th_opt=2,
                  specified=True), n_lbc_intervals=2))
    assert slots
    for slot in sorted(slots):
        kind = restart_mod.classify_scratch_slot(slot)
        assert kind in ("serialize", "rebuild"), (slot, kind)
    for slot in sorted(s for s in slots if s.startswith("mp_thompson_aero_")):
        assert restart_mod.classify_scratch_slot(slot) == "rebuild", slot
        assert slot not in restart_mod.SERIALIZED_SCRATCH_SLOTS


def test_mp28_smag_held_tendencies_cover_every_transported_species():
    from gpuwm.core import moist

    cfg = RunConfig(**_TINY, **_MP28, km_opt=4)
    slots = pf.scratch_slot_registry(cfg)
    import gpuwm.core.state as state_mod
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(state_mod, "cp", np)
        state = state_mod.DomainState(cfg)
    finally:
        monkey.undo()
    for name in moist.moist_species(state):
        assert "smag_r" + name in slots, name
    # Morrison's untransported nc keeps no held tendency.
    mp10 = pf.scratch_slot_registry(
        RunConfig(**_TINY, moist=True, mp_physics=10, km_opt=4))
    assert "smag_rnc" not in mp10


def test_mp28_nest_field_kinds_and_pricing():
    cfg = RunConfig(**_TINY, **_MP28)
    assert pf.nest_field_kinds(cfg) == (
        "u", "v", "w", "t", "ph", "mu",
        "qv", "qc", "qr", "qi", "qs", "qg",
        "nr", "ni", "nc", "nwfa", "nifa")
    # mp=10 stays exactly as ratified -- no nc.
    assert pf.nest_field_kinds(
        RunConfig(**_TINY, moist=True, mp_physics=10)) == (
        "u", "v", "w", "t", "ph", "mu",
        "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni", "ns", "ng")


def test_mp28_is_priced_and_never_falls_through_to_a_guess():
    """``domain_kernel_modules`` fails closed on an unpriced selector; mp=28
    must be a priced row, and it must name the modules the adapter really
    launches -- including the frozen mp=8 ``thompson`` module, whose
    sedimentation launchers mp=28 reuses byte-for-byte."""
    from datetime import datetime as _dt

    from gpuwm.experiment import experiment_from_run_config

    cfg = RunConfig(**_TINY, **_MP28, output_interval_s=1.0)
    exp = experiment_from_run_config(cfg, _dt(1974, 4, 3, 12))
    modules = pf.physics_kernel_modules(exp)
    for name in ("thompson", "thompson_aerosol_state", "thompson_aerosol_sat",
                 "thompson_aerosol_cold", "thompson_aerosol_warm",
                 "thompson_aerosol_sed"):
        assert name in modules, name
    # The probe translation unit is oracle-only and must never be priced
    # into a forecast's local-memory reservation.
    assert "thompson_aerosol_probe" not in modules
    # Every priced module has a driver-measured frame.
    frames = pf.kernel_local_frame_bytes(exp)
    for name in modules:
        assert name in frames, name
    assert pf.kernel_local_memory_bytes(exp) >= 0


# ---------------------------------------------------------------------------
# mp_physics=28 -- the PhysicsDriver budgets.
#
# Deliberately built from ``_TINY`` rather than from the four-domain flagship
# fixture: the flagship configs reference an external ERA5 forcing file that a
# clean checkout does not carry, so every test keyed on them ERRORS at
# collection and could not gate anything.  These three run anywhere.
# ---------------------------------------------------------------------------

_MP28_PBL = dict(bl_pbl_physics=1, sf_sfclay_physics=1)


def test_mp28_physics_driver_budget_admits_the_pbl_ice_tendency():
    """``pbl_tendencies/rqi`` is priced for mp=28 + YSU.

    BEFORE THIS TEST: ``preflight.py:1400`` read ``(6, 8, 10, 18)`` and an
    mp=28 + YSU domain's ``rqi`` stack was neither budgeted nor materialized,
    so the ``--alloc`` measurement understated that run's persistent driver
    set by one mass-grid array (two with a separate composed target).

    ``Registry/Registry.EM_COMMON:3036`` declares the ``thompsonaero``
    package as ``moist:qv,qc,qr,qi,qs,qg``, which is what makes WRF's
    ``F_QI`` true and ``module_first_rk_step_part1.F:1112``'s
    ``CALL pbl_driver`` pass ``moist(...,P_QI), F_QI=F_QI`` (:1199).
    """
    cfg = RunConfig(**_TINY, **_MP28, **_MP28_PBL)
    shapes = pf.physics_array_shapes(cfg)
    assert shapes["pbl_tendencies/rqi"] == (cfg.nz, cfg.ny, cfg.nx)


def test_the_pbl_rqi_budget_matches_the_runtime_predicate_for_every_scheme():
    """The budget and the runtime must not be able to disagree.

    ``preflight.physics_array_shapes`` restates the membership test that
    ``physics._pbl_optional_tendency_components`` decides at run time, and
    ``preflight._materialize_physics`` restates it a third time.  Two of the
    three were updated for mp=18 in an earlier wave and this file never
    checked the agreement; asserting it over every accepted selector is what
    stops the next scheme landing in two of three places.
    """
    from gpuwm.core.physics import _pbl_optional_tendency_components

    for mp in (0, 1, 6, 8, 10, 18, 28):
        cfg = RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=mp,
                        **_MP28_PBL)
        priced = "pbl_tendencies/rqi" in pf.physics_array_shapes(cfg)
        at_runtime = "rqi" in _pbl_optional_tendency_components(cfg)
        assert priced == at_runtime, (
            f"mp_physics={mp}: preflight prices pbl_tendencies/rqi="
            f"{priced} but physics.py composes rqi={at_runtime}")
    # And the PBL-off case prices none of it, for any scheme.
    assert "pbl_tendencies/rqi" not in pf.physics_array_shapes(
        RunConfig(**_TINY, **_MP28))


def test_mp28_driver_aliases_the_scheme_accumulators_instead_of_copies():
    """No private ``microphysics/*`` arrays for a scheme with a slot row.

    BEFORE THIS TEST: ``microphysics_scratch_slots(28)`` returned ``()``, so
    the mp=28 PhysicsDriver allocated three private zero-filled surface
    arrays (``microphysics/rainnc``, ``/rainncv``, ``/sr``) and
    ``accept_microphysics`` copied the scheme's result into them on every
    step, instead of aliasing the seven canonical ``mp_*`` scratch
    accumulators the aerosol adapter writes.

    The mp=0 control keeps its three: with no scheme there is no canonical
    set to alias, which is what the three arrays are for.
    """
    from gpuwm.core.physics import microphysics_scratch_slots

    shapes = pf.physics_array_shapes(RunConfig(**_TINY, **_MP28, **_MP28_PBL))
    assert not any(name.startswith("microphysics/") for name in shapes), (
        "mp=28 still budgets private driver-owned precipitation arrays")

    control = pf.physics_array_shapes(
        RunConfig(**_TINY, moist=True, mp_physics=0, **_MP28_PBL))
    assert {name for name in control if name.startswith("microphysics/")} == {
        "microphysics/rainnc", "microphysics/rainncv", "microphysics/sr"}

    slots = dict(microphysics_scratch_slots(28))
    assert slots == {
        "rainnc": "mp_rainnc", "rainncv": "mp_rainncv", "sr": "mp_sr",
        "snownc": "mp_snownc", "snowncv": "mp_snowncv",
        "graupelnc": "mp_graupelnc", "graupelncv": "mp_graupelncv"}
    # Every one of those slots is in the mp=28 scratch registry already, so
    # the aliasing adds no allocation anywhere -- it removes three.
    registry = pf.scratch_slot_registry(RunConfig(**_TINY, **_MP28),
                                        n_lbc_intervals=2)
    for slot in slots.values():
        assert slot in registry, slot


def test_mp28_rrtmgp_column_inventory_carries_the_effective_radii():
    """WRF's ``use_mp_re`` table lists THOMPSONAERO; the columns are priced.

    ``phys/module_physics_init.F:1005`` (THOMPSON) and ``:1006``
    (THOMPSONAERO) sit in the same disjunction, and the P3/Jensen-Ishmael
    ``has_reqs = 0`` override at ``:1026-1033`` does not touch either, so all
    three of ``has_reqc``/``has_reqi``/``has_reqs`` are 1 for mp=28.

    BEFORE THIS TEST: ``preflight.py:2708`` read ``(6, 8, 18)`` and an mp=28
    RTE+RRTMGP domain priced no radii columns at all.

    Thompson has no ``effr`` (that is Morrison's), and the legacy-RRTMG 4/4
    variant is priced as one shared call-peak envelope rather than through
    this function -- both asserted so a future edit cannot quietly widen the
    row into either.
    """
    cfg = RunConfig(**_TINY, **_MP28, ra_physics=4)
    shapes = pf.rrtmgp_column_shapes(cfg)
    ncol, nz = cfg.ny * cfg.nx, cfg.nz
    for name in ("effc", "effi", "effs"):
        assert shapes[f"columns/{name}"] == ((ncol, nz), 4), name
    assert "columns/effr" not in shapes
    assert "columns/nc" not in shapes

    # mp=8's row is untouched, and Morrison still gets its four radii.
    assert "columns/effc" in pf.rrtmgp_column_shapes(
        RunConfig(**_TINY, moist=True, mp_physics=8, ra_physics=4))
    assert "columns/effr" in pf.rrtmgp_column_shapes(
        RunConfig(**_TINY, moist=True, mp_physics=10, ra_physics=4))
    # Kessler declares no radii in WRF's table and must not price any.
    assert not any("eff" in name for name in pf.rrtmgp_column_shapes(
        RunConfig(**_TINY, moist=True, mp_physics=1, ra_physics=4)))


# ---------------------------------------------------------------------------
# mp_physics=28 -- the rest of the WP-10 infrastructure surface.
#
# These are not preflight tests.  They live here because tests/test_preflight
# .py is the ONE test file WP-10 owns, and the alternative is shipping the
# state/transport/restart/nesting/pricing work with no regression coverage at
# all.  Each one names the module it actually guards.
# ---------------------------------------------------------------------------

def _host_mp28_state(**overrides):
    """A real ``DomainState`` on numpy, so no device is required."""
    import gpuwm.core.state as state_mod

    cfg = RunConfig(**_TINY, **{**_MP28, **overrides})
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(state_mod, "cp", np)
        return state_mod.DomainState(cfg), cfg
    finally:
        monkey.undo()


def test_mp28_acoustic_cq_sums_six_masses_and_no_number_moment():
    """gpuwm/core/acoustic.py.  WRF's calc_cq sums the Registry ``moist``
    package only; mp=28's qnc/qnwfa/qnifa are ``scalar``.  A droplet number
    of order 1e8 leaking into q_tot would not be subtle, but n_mass is an
    integer passed to a kernel and a wrong value is invisible from Python.
    """
    from gpuwm.core import acoustic

    captured = {}

    def fake_get_kernel(module, func):
        assert (module, func) == ("acoustic", "calc_cq")

        def launch(_grid, _block, args):
            captured["n_mass"] = int(args[10])
            captured["fields"] = args[:7]
        return launch

    state, cfg = _host_mp28_state()
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(acoustic, "get_kernel", fake_get_kernel)
        monkey.setattr(acoustic, "cp", np, raising=False)
        _, _, _, use = acoustic.prepare_moist_cq(state, cfg)
    finally:
        monkey.undo()
    assert use is True
    # Identical to mp=8: qv, qc, qr, qi, qs, qg.
    assert captured["n_mass"] == 6
    # And the seventh slot (qh) is the qv placeholder, never an aerosol.
    assert captured["fields"][6] is state.qv
    for moment in (state.nc, state.nwfa, state.nifa):
        assert not any(moment is arg for arg in captured["fields"])


def test_mp28_npref_cq_species_match_mp8_exactly():
    """gpuwm/verify/npref.py -- the CPU mirror must make the same choice."""
    from gpuwm.verify import npref

    assert (npref._CQ_MASS_SPECIES_BY_MP[28]
            == npref._CQ_MASS_SPECIES_BY_MP[8])
    moisture = {name: np.full((2, 2, 2), 0.001) for name in
                ("qv", "qc", "qr", "qi", "qs", "qg")}
    # Adding aerosol fields to the dict must not change the answer.
    polluted = dict(moisture, nc=np.full((2, 2, 2), 1.0e8),
                    nwfa=np.full((2, 2, 2), 3.0e8),
                    nifa=np.full((2, 2, 2), 5.0e3))
    for clean, dirty in zip(npref.np_calc_cq(moisture, 28),
                            npref.np_calc_cq(polluted, 28), strict=True):
        np.testing.assert_array_equal(np.asarray(clean), np.asarray(dirty))


def test_mp28_health_rules_cover_the_aerosol_tracers():
    """gpuwm/core/health.py.  An uncovered field is a field the integration
    health gate silently ignores -- the exact failure mode this port is most
    exposed to, since a wrong aerosol number stays finite and bounded."""
    from gpuwm.core import health

    state, _ = _host_mp28_state()
    names = {f.name for f in health.collect_state_fields(state)}
    for name in ("nc", "nwfa", "nifa", "nr", "ni", "qi", "qs", "qg"):
        assert name in names, name
    for leaf in ("nwfa", "nifa"):
        rule = health.rule_for_field(leaf)
        assert rule.status_class == "moment"
        assert rule.lower == 0.0
        # WRF's own terminal ceiling is 9999.E6 with an unclamped surface
        # emission on top (module_mp_thompson.F:3977-3982, :1310-1327), so
        # the rule must sit well above it without being unbounded.
        assert rule.upper >= 9999.0e6
    # The census must not have started covering Morrison's untransported nc
    # differently, and must not have picked up the 2-D emission fields
    # (those are constants, not integration state).
    assert "nwfa2d" not in names and "nifa2d" not in names


def test_mp28_lateral_boundary_allow_lists_accept_the_new_scalars():
    """gpuwm/ingest/lateral_bc.py -- the three allow-lists."""
    import inspect

    from gpuwm.ingest import lateral_bc

    for func in (lateral_bc.apply_specified_relaxation,
                 lateral_bc.couple_nest_field,
                 lateral_bc.uncouple_feedback_field):
        source = inspect.getsource(func)
        for name in ("nc", "nwfa", "nifa"):
            assert f'"{name}"' in source, (func.__name__, name)


def test_mp28_external_lbc_carries_only_qv_and_says_so():
    """The registered deviation, pinned so it cannot be un-registered by
    accident.  ArWen gives every non-qv scalar a flow-dependent boundary with
    ZERO inflow; for aerosols that monotonically depletes nwfa/nifa in the
    upstream boundary zone, which WRF does not do (its Registry gives
    qnwfa/qnifa real bdy arrays).  If someone extends
    ``_coupled_device_fields``, this test fails and the prose that explains
    the deviation must be retired in the same diff."""
    import inspect

    from gpuwm.core import moist
    from gpuwm.ingest import lateral_bc

    source = inspect.getsource(lateral_bc._coupled_device_fields)
    coupled = {line.split('"')[1] for line in source.splitlines()
               if line.strip().startswith('result["')
               or line.strip().startswith('"')}
    for name in ("nwfa", "nifa", "nc", "qc", "qr", "qi", "qs", "qg"):
        assert name not in coupled, name
    assert "qv" in coupled
    # The deviation is written down where a reader will find it.
    assert "ZERO AEROSOL INFLOW" in moist.__doc__
    assert "REGISTERED DEVIATION" in source


def test_mp28_mixed_nest_edge_is_refused_by_name():
    """gpuwm/core/microphysics_transition.py.

    v1 refuses rather than inventing an entry closure for nc/nwfa/nifa
    across a scheme boundary.  The refusal must (a) fire for BOTH
    directions and every partner scheme, (b) name mp=28 and its moments
    rather than reading as "mp=28 is not implemented", and (c) NOT touch
    the same-scheme mp28 -> mp28 nest, which is a supported configuration.
    """
    import types

    from gpuwm.core import microphysics_transition as mt

    def run(mp, policy=mt.SAME_SCHEME_POLICY):
        return types.SimpleNamespace(
            mp_physics=mp, moist=True, moist_cq=True,
            nest_microphysics_transition=policy,
            morr_rimed_ice=1, wsm6_hail_opt=0)

    same = mt.resolve_microphysics_transition(run(28), run(28))
    assert same.mixed is False
    assert same.policy_id == mt.SAME_SCHEME_POLICY

    partners = [mp for mp in mt.PORTED_MP_PHYSICS]
    assert partners, "PORTED_MP_PHYSICS became empty"
    for other in partners:
        for parent, child in ((other, 28), (28, other)):
            with pytest.raises(ValueError) as excinfo:
                mt.resolve_microphysics_transition(
                    run(parent), run(child, mt.EDGE_MATRIX_POLICY))
            message = str(excinfo.value)
            assert "REFUSED" in message
            assert "MP28" in message
            for moment in ("nc", "nwfa", "nifa"):
                assert moment in message, (parent, child, moment)
            # It must not masquerade as "scheme not ported".
            assert "ported selectors are" not in message

    # And the ratified MP8 -> MP18 edge codes are untouched.
    assert mt._EDGE_FIELD_CODES["qvolh"] == 19
    assert len(mt._EDGE_FIELD_CODES) == 20
    assert 28 not in mt.PORTED_MP_PHYSICS


def test_mp28_survives_a_restart_round_trip(tmp_path):
    """gpuwm/io/restart.py + gpuwm/state_serialization_contract.py.

    Every mp=28 field must be classified, written, and restored bit-for-bit
    -- including the two 2-D surface emission constants, which nothing in the
    forecast writes and which would therefore come back as zeros (silently
    switching off surface aerosol emission for the rest of the run) if they
    were merely 'rebuilt'.
    """
    import gpuwm.core.state as state_mod
    from gpuwm.io import restart as restart_mod

    state, cfg = _host_mp28_state()
    rng = np.random.default_rng(28)
    written = {}
    for name in ("nc", "nr", "ni", "nwfa", "nifa", "qc", "qi",
                 "nwfa2d", "nifa2d"):
        array = getattr(state, name)
        array[...] = rng.random(array.shape).astype(np.float32) * 1.0e8
        written[name] = array.copy()

    manifest = restart_mod.state_manifest(state)
    for name in written:
        assert f"state/{name}" in manifest, name
    # No mp=28 attribute may be unclassified -- state_manifest walks every
    # instance attribute through classify_state_attr and raises otherwise,
    # so reaching this line already proves it, but name the new ones.
    for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
        assert restart_mod.classify_state_attr(name) == "serialize"
    for name in ("nc0", "nwfa0", "nifa0"):
        assert restart_mod.classify_state_attr(name) == "rebuild"

    # Restore into a fresh state and compare.
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(state_mod, "cp", np)
        restored = state_mod.DomainState(cfg)
    finally:
        monkey.undo()
    for name, value in written.items():
        assert not np.array_equal(getattr(restored, name), value)
        getattr(restored, name)[...] = manifest[f"state/{name}"]
        np.testing.assert_array_equal(getattr(restored, name), value)


def test_mp28_nest_init_interpolates_and_seeds_every_new_field():
    """gpuwm/ingest/nest_init.py -- the interpolation list and the RK seed
    list.  A field missing from the seed list starts its first child RK step
    with a zero time-t copy, which is a one-step transient no bound catches.
    """
    import inspect

    from gpuwm.ingest import nest_init

    source = inspect.getsource(nest_init)
    for entry in ('("nwfa", "")', '("nifa", "")',
                  '("nwfa2d", "")', '("nifa2d", "")',
                  '("nc", "nc0")', '("nwfa", "nwfa0")',
                  '("nifa", "nifa0")'):
        assert entry in source, entry

    # The RK seed loop is None-guarded, so a Morrison child (nc present,
    # nc0 absent) must be unaffected by the new ("nc", "nc0") row.
    import gpuwm.core.state as state_mod
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(state_mod, "cp", np)
        mp10 = state_mod.DomainState(
            RunConfig(**_TINY, moist=True, mp_physics=10))
    finally:
        monkey.undo()
    assert getattr(mp10, "nc", None) is not None
    assert getattr(mp10, "nc0", None) is None


@pytest.mark.gpu
def test_mp28_scalars_actually_advect_on_the_device():
    """The transport claim, run rather than asserted.

    ``extra_moist_species`` returning the right tuple only proves the loop
    would VISIT nc/nwfa/nifa.  This drives the real positive-definite
    advection stage on the GPU with a uniform x-flow at face Courant 0.5 and
    requires that each of the five mp=28 number moments (a) moves, (b) stays
    finite, (c) stays non-negative, and (d) conserves its coupled mass under
    the periodic flux telescope.  A field that silently never entered the
    stage loop would sit unchanged and fail (a).
    """
    cp = pytest.importorskip("cupy")

    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import (
        advance_scalars_stage, extra_moist_species, init_moist_balanced)

    nx, ny, nz = 16, 8, 12
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=500.0, dy=500.0, ztop=6000.0,
                    dt=10.0, run_seconds=0.0, moist=True, mp_physics=28)
    vc = make_vertical_coord(nz)
    base = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, vc, base, lambda z: np.full(nz, 1.0e-3))

    assert extra_moist_species(state) == (
        "qi", "qs", "qg", "nr", "ni", "nc", "nwfa", "nifa")

    # Plausible magnitudes, deliberately spanning nine decades so an FP32
    # transport bug in the aerosol fields cannot hide behind qv's scale.
    blobs = {"nc": 1.0e8, "nwfa": 3.0e8, "nifa": 5.0e3,
             "nr": 1.0e4, "ni": 1.0e5}
    for name, amplitude in blobs.items():
        host = np.zeros((nz, ny, nx), dtype=np.float32)
        host[4:8, 3:6, 6:10] = amplitude
        getattr(state, name)[...] = cp.asarray(host)
        getattr(state, name + "0")[...] = getattr(state, name)

    chm = (state.c1h[:, None, None] * state.total_mu()[None]
           + state.c2h[:, None, None])
    dt_eff = cfg.dt
    ru = cp.zeros((nz, ny, nx + 1), cp.float32)
    ru[...] = 0.5 * (cfg.dx / dt_eff) * chm[:, :, :1]
    rv = cp.zeros((nz, ny + 1, nx), cp.float32)
    ww = cp.zeros((nz + 1, ny, nx), cp.float32)

    dnw_abs = -state.dnw[:, None, None]

    def coupled_mass(field):
        return float(cp.sum((chm * field * dnw_abs).astype(cp.float64)))

    before = {name: coupled_mass(getattr(state, name)) for name in blobs}
    advance_scalars_stage(state, cfg, ru, rv, ww, dt_eff, final=True)
    cp.cuda.Stream.null.synchronize()

    for name, amplitude in blobs.items():
        field = getattr(state, name)
        assert bool(cp.isfinite(field).all()), name
        assert float(field.min()) >= 0.0, name
        # (a) it MOVED: the blob's leading edge has advanced in +x.
        moved = float(cp.abs(field - getattr(state, name + "0")).max())
        assert moved > 0.01 * amplitude, (
            f"{name} did not advect -- it is allocated but not transported")
        # (d) coupled mass is conserved by the periodic telescope.
        residual = abs(coupled_mass(field) - before[name]) / before[name]
        assert residual < 1e-5, (name, residual)


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
    # Physics carries the domain's OLR publication buffer, one resident
    # (ny, nx) FP32 field: 4 * 200 * 250 = 200,000 B.
    assert by_cat == {
        "state": 563557756,
        "physics": 275906760,
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
    assert d01.resident_bytes == 1473842400
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
    # OLR publication buffer: one resident (ny, nx) FP32 field per 4/4
    # domain, so each domain gains exactly 4*ny*nx B -- 200,000 /
    # 800,000 / 1,004,004 / 1,440,000 on d01--d04 (sum 3,444,004 B).
    assert per_domain == {1: 1503530600, 2: 5403918992,
                          3: 6800616700, 4: 9730339968}
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
    # residency plus the exact 23,815,680-B ring total, plus the
    # 3,444,004-B four-domain OLR publication total.
    assert est4.resident_bytes == 14436554968
    # The case configures column_chunk = 6250 (byte-identical to 3125,
    # 33% faster per radiation call); the 3125 numbers stay pinned in the
    # ladder below, so the trade this bought is on the record both ways.
    assert est4.workspace_bytes == 2051400000
    assert est4.transient_peak_bytes == 3182840000
    # +3,444,004 B: the four-domain OLR publication total.
    assert est4.subtotal_bytes == 19670794968
    assert est4.alloc_estimate_bytes == math.ceil(
        1.15 * est4.subtotal_bytes) == 22621414214
    # Chunk ladder after arena sharing and physics-persistent reclamation.
    # The 1024-descriptor health-slot registration adds 49,168 B/domain to
    # the audited scratch; the pins below are computed on the merged tree.
    ladder = {chunk: pf.estimate_experiment(
        exp4, column_chunk=chunk).alloc_estimate_bytes
        for chunk in (6250, 3125, 1562, 256)}
    # Every rung carries the ring lane's ceil(1.15 x 23,815,680) =
    # 27,388,032 B on top of the ports-branch ladder.
    # Every rung also carries ceil(1.15 x 3,444,004) of OLR.
    assert ladder == {6250: 22621414214, 3125: 21425874214,
                      1562: 20827912927, 256: 20328272850}


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
    (health descriptors, nested-force slots) and the OLR publication
    buffers (x1.15 headroom = 3,960,605 B of estimate) the stale
    projection now sits 5,277,330 B BELOW the estimate and the
    measured-bound leg reads True.  The
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
    assert projected_used - est.alloc_estimate_bytes == -5_277_330
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
    # lands 27,388,032 B above the ports-branch CLI pin; the OLR
    # publication buffers add a further 3,960,605 B of estimate.
    assert payload["alloc_estimate_bytes"] == 22621414214
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
    # ring lane's 27,388,032 B: +821,641 B over the ports-branch pin,
    # and 3% of the OLR estimate is a further +118,818 B.
    assert payload["reserve_bytes"] == 3811771131
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

    # And end to end through the CLI: a declared budget close enough to
    # the card that adding the reserve back overshoots its capacity.
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "15",
                     "--vram-gib", "16", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["reserve_bytes"] > gib, "otherwise nothing to cap"
    assert payload["measured_free_bytes"] <= 16 * gib
    assert payload["free_bytes_capped_to_physical_bytes"] is not None
    assert "capped" in payload["free_bytes_source"]
    # The budget follows the capped free, so the gate is evaluated
    # against VRAM that exists.
    assert payload["budget_bytes"] == (
        payload["measured_free_bytes"] - payload["reserve_bytes"])
    assert rc != 0, "22.6 GiB of estimate does not fit a 16 GB card"

    # Text mode says so out loud rather than only in --json.
    _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "15",
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
    # The WDDM lane keeps its one instrumented multiplicative observation
    # as a FLOOR: the affine form may raise it, never discount it.
    assert payload["observed_peak_envelope_bytes"] >= int(
        payload["footprint_projection_bytes"] * 1.75)
    # 100 GiB budget: envelope fits, no warning.
    assert payload["observed_peak_envelope_exceeds_budget"] is False
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "keeps that multiplier as a FLOOR" in out
    assert "WARNING: observed peak envelope" not in out
    # The forecast is no longer the only phase this report prices, so the
    # historical line must say which phase it is, the preprocessing phase
    # must appear beside it, and one sentence must name the binding one.
    assert "FORECAST PEAK ENVELOPE" in out
    assert "INGEST OBSERVED PEAK ENVELOPE" in out
    assert "INGEST (preprocessing, --source era5)" in out
    assert "BINDING PHASE:" in out
    assert "memory-binding phase" in out

    # A budget the ESTIMATE fits but the envelope exceeds: the estimate
    # gate still passes, and the warning names the honest number.
    envelope_gib = payload["peak_envelope_bytes"] / GIB
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
    # AFFINE envelope over it clears every instrumented run -- the three
    # 2026-07-30 pilots, each re-read with the non-pool term its own card
    # carries rather than the 5090's.
    pilots = ((7.20, 9.54, 128), (7.29, 8.99, 128), (3.51, 4.04, 46))
    for alloc_gib, measured_gib, sms in pilots:
        profile = pf.DeviceLocalMemoryProfile(
            name="pilot", multiprocessor_count=sms,
            max_threads_per_multiprocessor=1536)
        non_pool = pf.CUDA_CONTEXT_BYTES + profile.reservation_bytes(
            pf.LEVEL_SPECIALIZED_KERNEL_FRAMES["kf"].frame_bytes(49))
        envelope = pf.machine_peak_envelope_bytes(
            alloc_estimate_bytes=int(alloc_gib * GIB),
            non_pool_bytes=non_pool, family="linux")
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
    # AFFINE on Linux: estimate + the itemized non-pool residency + the
    # measured unmodelled constant (+ a per-nest term).  Not a multiple
    # of the projection -- a multiple has no intercept, and a model with
    # no intercept changes the SIGN of its error with grid size.
    assert payload["observed_peak_envelope_bytes"] == (
        payload["alloc_estimate_bytes"]
        + payload["non_pool_device_bytes"]
        + pf.ENVELOPE_UNMODELLED_BYTES
        + math.ceil(pf.ENVELOPE_PER_NEST_FRACTION
                    * (len(payload["domains"]) - 1)
                    * payload["alloc_estimate_bytes"]))
    # And the projection itself dropped the two Windows-pool constants.
    assert payload["footprint_projection_bytes"] == payload[
        "alloc_estimate_bytes"]
    assert payload["reserve_components"]["retention_residual_bytes"] >= 0

    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "FORECAST PEAK ENVELOPE (estimate" in out
    assert "affine, not a multiplier" in out
    assert "1.746x its footprint projection" not in out
    assert "INGEST OBSERVED PEAK ENVELOPE" in out
    assert "BINDING PHASE:" in out


def test_check_cli_legacy_config_wraps(capsys):
    rc = _run_check(["check", str(CONFIG_D01), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert list(payload["domains"]) == ["d01"]
    # The 1024-descriptor health capacity adds 24,576 B and the ring-guard
    # saves 3,010,560 B to the pre-assembly pin, and the OLR publication
    # buffer a further 200,000 B (itemization-pin derivation).
    assert payload["domains"]["d01"]["resident_bytes"] == 1473842400


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


# ---------------------------------------------------------------------------
# 2026-08-01 sizing calibration: the affine envelope and the tree ingest
# ---------------------------------------------------------------------------

#: Every whole-forecast run instrumented on the 16 GiB fleet node (RTX
#: 4080, Linux, driver 595.58.03, machine-wide ``nvidia-smi`` at 250 ms,
#: GPU otherwise idle), as ``(label, domains, itemized alloc estimate GiB,
#: measured machine peak GiB)``.  The 4080 carries 76 SMs, which is what
#: sizes the non-pool term for every row.
FLEET_4080_FORECAST_RUNS = (
    ("s07      170x136",              1,  2.0652,  3.6494),
    ("small8   224x180",              1,  2.7500,  4.1436),
    ("small8   224x180 (go route)",   1,  2.7500,  4.3818),
    ("s11      340x272",              1,  4.8242,  5.9541),
    ("edge15   448x360 (go route)",   1,  7.5576,  8.7529),
    ("L12      474x378 (go route)",   1,  8.2683,  9.2510),
    ("over22   594x476 (go route)",   1, 12.3809, 12.5889),
    ("big24    630x504 (go route)",   1, 13.7616, 13.8799),
    ("n10      2-domain tree",        2,  4.1204,  5.8330),
    ("n2_16    2-domain tree",        2,  8.2162, 10.0928),
    ("c07      4-domain tree",        4,  2.0612,  3.8564),
)

#: SM count of the card every row above was measured on.
FLEET_4080_MULTIPROCESSORS = 76


def _fleet_4080_non_pool_bytes(nz: int = 49) -> int:
    """The non-pool term a 4080 carries for the default suite at ``nz``."""

    profile = pf.DeviceLocalMemoryProfile(
        name="RTX 4080", multiprocessor_count=FLEET_4080_MULTIPROCESSORS,
        max_threads_per_multiprocessor=1536)
    widest = pf.LEVEL_SPECIALIZED_KERNEL_FRAMES["kf"].frame_bytes(nz)
    return pf.CUDA_CONTEXT_BYTES + profile.reservation_bytes(widest)


def test_the_envelope_bounds_every_instrumented_run():
    """An envelope must never land under a measured peak.

    The x1.45 multiplier did, because it had no intercept: the smallest
    run in this table was declared 3.99 GiB and peaked at 4.38.  The same
    model over-predicted the largest by 30%.  One model, no intercept,
    read at two grid sizes -- which is also why a 5090 datapoint saying
    "19% under" and this card saying "25-30% over" were never in
    conflict.
    """
    non_pool = _fleet_4080_non_pool_bytes()
    worst_over = 0.0
    for label, domains, estimate_gib, measured_gib in (
            FLEET_4080_FORECAST_RUNS):
        envelope = pf.machine_peak_envelope_bytes(
            alloc_estimate_bytes=int(estimate_gib * GIB),
            non_pool_bytes=non_pool, domains=domains, family="linux")
        assert envelope >= measured_gib * GIB, label
        over = envelope / (measured_gib * GIB) - 1.0
        worst_over = max(worst_over, over)
    # Conservative, but not absurdly so: the old model reached +30% at
    # the top of this table while being optimistic at the bottom.
    assert worst_over < 0.30


def test_the_old_multiplier_is_optimistic_where_the_affine_form_is_not():
    """The negative control for the test above.

    This is the defect, executed: the retired multiplicative envelope
    lands UNDER the measured peak of the smallest run in the table, and
    the affine one does not.  If this ever stops failing for the old
    model, the evidence changed and the calibration needs re-deriving.
    """
    label, domains, estimate_gib, measured_gib = FLEET_4080_FORECAST_RUNS[1]
    footprint = int(estimate_gib * GIB)  # Linux: projection == estimate
    old = pf.observed_peak_envelope_bytes(footprint, platform="linux")
    assert old < measured_gib * GIB, (
        "the retired x1.45 envelope must still be the optimistic one "
        "this calibration replaced")
    new = pf.machine_peak_envelope_bytes(
        alloc_estimate_bytes=footprint,
        non_pool_bytes=_fleet_4080_non_pool_bytes(), domains=domains,
        family="linux")
    assert new >= measured_gib * GIB


def test_the_envelope_has_an_intercept_that_does_not_scale_with_the_grid():
    """The structural property, not a number: doubling the estimate must
    NOT double the envelope, because part of it is a device constant."""

    non_pool = _fleet_4080_non_pool_bytes()
    small = pf.machine_peak_envelope_bytes(
        alloc_estimate_bytes=2 * GIB, non_pool_bytes=non_pool,
        family="linux")
    large = pf.machine_peak_envelope_bytes(
        alloc_estimate_bytes=4 * GIB, non_pool_bytes=non_pool,
        family="linux")
    assert large - small == 2 * GIB, "the pool side is 1:1"
    assert large < 2 * small, "an intercept is not a multiplier"
    # And the intercept is the thing this module already itemizes.
    assert small - 2 * GIB == non_pool + pf.ENVELOPE_UNMODELLED_BYTES


def test_the_wddm_multiplier_is_a_floor_not_a_discount():
    """Windows keeps its one instrumented observation.  The affine form
    may raise that number; it may never lower it."""

    footprint = 20 * GIB
    windows = pf.machine_peak_envelope_bytes(
        alloc_estimate_bytes=8 * GIB, non_pool_bytes=GIB,
        footprint_projection_bytes=footprint, family="windows")
    assert windows == int(footprint * pf.PEAK_ENVELOPE_FACTORS["windows"])
    # ...and where the affine form is the larger of the two, it wins.
    huge = pf.machine_peak_envelope_bytes(
        alloc_estimate_bytes=40 * GIB, non_pool_bytes=GIB,
        footprint_projection_bytes=footprint, family="windows")
    assert huge > int(footprint * pf.PEAK_ENVELOPE_FACTORS["windows"])


def test_the_absent_card_profile_is_the_conservative_measured_reference():
    """Sizing a card that is NOT in this machine never discounts the
    intercept below what a real device measures.

    The 4090 stress run (2026-08-03) falsified the per-class SM discount:
    an absent-card sizing priced the non-pool intercept at 1.45 GiB (the
    12 GiB class's 70-SM row) where the same code on the real RTX 4090
    measured 2.30 GiB -- a config certified "fits with 0.27 GiB to
    spare" landed 0.015 GiB from the budget, a margin 18x smaller than
    advertised.  The class table was a market survey, not a measurement
    (a 12 GiB RTX 3080 Ti ships 80 SMs against the row's 70), so the
    absent-card path now prices against the conservative measured
    reference profile -- the max of known-device intercepts.
    """
    for gib in (12.0, 16.0, 24.0, 32.0, None, 999.0):
        assert (pf.card_local_memory_profile(gib)
                is pf.MEASURED_LOCAL_MEMORY_PROFILE), gib


def test_absent_card_sizing_is_never_more_optimistic_than_a_present_card(
        exp4):
    """THE stress-run inequality: for the same config, the absent-card
    estimate must be >= the present-card measurement, for every device
    this project has measured the law on."""
    known_devices = (
        pf.MEASURED_LOCAL_MEMORY_PROFILE,  # RTX 5090, 170 SMs
        pf.DeviceLocalMemoryProfile(       # the stress run's RTX 4090
            name="NVIDIA GeForce RTX 4090", multiprocessor_count=128,
            max_threads_per_multiprocessor=1536),
        pf.DeviceLocalMemoryProfile(       # the fleet's RTX 4080
            name="NVIDIA GeForce RTX 4080", multiprocessor_count=76,
            max_threads_per_multiprocessor=1536),
    )
    for gib in (12.0, 16.0, 24.0, 32.0):
        absent = pf.non_pool_device_bytes(
            exp4, profile=pf.card_local_memory_profile(gib))
        for device in known_devices:
            present = pf.non_pool_device_bytes(exp4, profile=device)
            assert absent >= present, (
                f"sizing an absent {gib:g} GiB card priced the non-pool "
                f"intercept at {absent / GIB:.2f} GiB, below the "
                f"{present / GIB:.2f} GiB the same config prices on a "
                f"present {device.name}")


def test_declared_budget_sizing_says_it_is_an_estimate_for_absent_hardware(
        capsys):
    """--budget-gib is the sizing-for-a-card-you-intend-to-buy path; its
    report must say the numbers are estimates for hardware not present."""
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "HARDWARE NOT PRESENT" in out
    assert "declared, not measured" in out

    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "100",
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["sized_for_hardware_not_present"] is True
    assert payload["local_memory_profile"] == (
        pf.MEASURED_LOCAL_MEMORY_PROFILE.name)


def test_ingest_prices_every_domain_in_the_tree(exp1, exp4):
    """v1.4.0 priced this phase on the ROOT alone.

    So the prediction FELL as nests were added -- 5.46 -> 1.89 -> 1.30
    GiB across one, two and four domains -- while the machine measured it
    flat at 4.0-4.8, under by 3.4x at four domains and in the unsafe
    direction, on the very number the before-the-fetch gate half-relies
    on.  A deeper ladder has a smaller root; only the root was priced;
    so adding domains made the answer shrink.
    """
    one = pf.estimate_ingest(exp1, source="gfs")
    four = pf.estimate_ingest(exp4, source="gfs")

    assert one.nest_state_bytes == 0, "a single domain has no nests"
    assert one.nest_state_items == ()
    assert four.nest_state_bytes > 0
    assert len(four.nest_state_items) == len(exp4.domains) - 1

    # Every nest carries one complete initial state, and they are all
    # resident for the single export transaction.
    assert four.resident_bytes == (
        four.resident_times * four.per_time_bytes
        + four.forcing_table_bytes + four.nest_state_bytes)

    # The transient is charged against the WIDEST domain in the tree, not
    # the root: on a real ladder the widest domain is usually a nest.
    assert four.transient_basis_bytes >= four.per_time_bytes

    # THE DEFECT, as an inequality: the tree's ingest estimate must not
    # be reachable by pricing the root alone.
    root_only = (math.ceil(one.headroom * (
        four.resident_times * four.per_time_bytes
        + four.forcing_table_bytes
        + math.ceil(pf.INGEST_TRANSIENT_PER_TIME_FRACTION
                    * four.per_time_bytes)))
        + four.context_bytes + four.device_overhead_bytes)
    assert four.peak_envelope_bytes > root_only


def test_adding_a_nest_never_lowers_the_ingest_estimate():
    """The falsifiable form of the same defect.

    Same root, one nest added: the answer must go UP.  Under the root-only
    model a two-domain tree whose root was smaller than the single-domain
    layout priced LOWER than the single domain, which is how the four-
    domain ladder came to declare 1.30 GiB and measure 4.39.
    """
    root = RunConfig(nx=240, ny=192, nz=49, dx=12000.0, dy=12000.0,
                     ztop=20000.0, dt=60.0, run_seconds=7200.0,
                     mp_physics=8, moist=True, terrain_opt=1,
                     spec_bdy_width=5, specified=True)
    alone = experiment_from_run_config(root, datetime(2026, 7, 30))
    one = pf.estimate_ingest(alone, source="gfs")

    child = dataclasses.replace(root, grid_id=2, nx=480, ny=384,
                                dx=3000.0, dy=3000.0, dt=15.0)
    from gpuwm.experiment import DomainConfig, ExperimentConfig
    tree = dataclasses.replace(
        alone, domains=alone.domains + (DomainConfig(
            grid_id=2, parent_id=1, i_parent_start=31, j_parent_start=25,
            parent_grid_ratio=4, parent_time_step_ratio=4,
            history_interval_s=3600.0, run=child),))
    two = pf.estimate_ingest(tree, source="gfs")

    assert two.peak_envelope_bytes > one.peak_envelope_bytes
    # And the nest that was 4.9x the root's cells is visible by name.
    assert [grid for grid, _ in two.nest_state_items] == [2]
    assert two.nest_state_bytes > one.per_time_bytes


def test_a_negative_budget_is_clamped_and_explained(capsys):
    """A reserve larger than free VRAM leaves NO budget.

    ``budget = free - reserve`` is unbounded below, and a 4000x4000
    config drove it to -7.15 GiB, which the report then printed as a
    capacity to compare an envelope against.
    """
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "0.001",
                     "--vram-gib", "1", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["budget_bytes"] >= 0, "a capacity is never negative"
    assert payload["budget_underwater_bytes"] > 0
    assert rc != 0

    _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "0.001",
                "--vram-gib", "1"])
    out = capsys.readouterr().out
    assert "NO BUDGET AT ALL" in out
    assert "the reserve alone is" in out
    assert " -" not in out.split("NO BUDGET AT ALL")[1].split("\n")[0]


def test_the_over_budget_remedy_is_an_action_not_a_design_pointer(capsys):
    """It used to end "staged residency (DESIGN REOPEN) per section E".

    No pip user has a section E, and the sentence names nothing to do.
    The actionable form already existed one layer up, in `gpuwm go`'s
    refusal, and is reused here.
    """
    _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "1",
                "--vram-gib", "32"])
    out = capsys.readouterr().out
    assert "OVER BUDGET" in out
    assert "DESIGN REOPEN" not in out
    assert "section E" not in out
    assert "remedy: re-size for this card -- gpuwm domain --vram-gib" in out


def test_the_printed_exit_code_is_the_one_the_process_returns(capsys):
    """The WARNING used to assert "(exit code 4: gates passed)" even when
    a gate had just failed and the process therefore exited 1."""

    # Envelope over, gates over too: the harder verdict, announced.
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "1",
                     "--vram-gib", "32"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "exit code 1: a gate above FAILED as well" in out
    assert "exit code 4: gates passed" not in out

    # Envelope over, every gate passed: 4, and it says 4.
    payload_rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib",
                             "100", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload_rc == 0
    tight = f"{payload['alloc_estimate_bytes'] / GIB + 0.5:.2f}"
    rc = _run_check(["check", str(CONFIG_4DOM), "--budget-gib", tight])
    out = capsys.readouterr().out
    assert rc == 4, out
    assert "exit code 4: gates passed, envelope did not" in out


def test_the_budget_word_follows_the_platform(capsys, monkeypatch):
    """"WDDM budget" on a Linux box, in the same report that has just
    finished explaining there is no WDDM here."""

    monkeypatch.setattr(pf.sys, "platform", "linux")
    _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "1",
                "--vram-gib", "32"])
    out = capsys.readouterr().out
    assert "WARNING: observed peak envelope" in out
    assert "exceeds the WDDM budget" not in out

    monkeypatch.setattr(pf.sys, "platform", "win32")
    _run_check(["check", str(CONFIG_4DOM), "--budget-gib", "1",
                "--vram-gib", "32"])
    out = capsys.readouterr().out
    assert "exceeds the WDDM budget" in out


def test_the_unmeasured_small_windows_tier_keeps_its_fixed_term():
    """Replacing a multiplier with an intercept must not drop a term.

    The Windows small-card tier is EXPERIMENTAL and calibrated on
    nothing: it stands in for the WDDM residency the CuPy pool never
    sees with one reduced fixed reserve.  The affine envelope carries
    that term in its intercept, so the tier does not quietly become more
    optimistic on the strength of Linux measurements.
    """
    assert pf.platform_projection_constants("win32", 12.0) == (
        0, pf.WINDOWS_SMALL_CARD_RESERVE_BYTES)
    exp = load_experiment_case(CONFIG_4DOM)[0]
    small = pf.estimate_experiment(exp, vram_gib=12.0)
    linux = dataclasses.replace(small, device_overhead_bytes=0,
                                envelope_family="linux")
    assert small.envelope_intercept_bytes - linux.envelope_intercept_bytes \
        == pf.WINDOWS_SMALL_CARD_RESERVE_BYTES
    assert small.peak_envelope_bytes > linux.peak_envelope_bytes
    # Linux itself carries no such term: the whole point of the 2026-07-30
    # amendment was that those constants are Windows-pool artifacts.
    on_linux = pf.estimate_experiment(exp, vram_gib=24.0)
    if pf.envelope_platform(vram_gib=24.0) == "linux":
        assert on_linux.envelope_intercept_bytes == \
            on_linux.non_pool_device_bytes


def test_the_gate_names_printed_on_linux_do_not_say_wddm(monkeypatch):
    """A-6.  WDDM is a Windows display driver model, and these three gate
    names are the first `gpuwm check` output a new user reads.  The prose
    beside them has been platform-correct since envelope_platform was
    introduced; only the names were left behind."""
    from gpuwm.core import preflight

    monkeypatch.setattr(preflight.sys, "platform", "linux")
    shown = [preflight.gate_display_name(m)
             for m in preflight.N0_GATE_METRICS]
    assert shown == ["alloc_fits_vram_budget", "alloc_measured_le_estimate",
                     "alloc_estimate_le_vram_budget"]
    assert not any("wddm" in name for name in shown)

    monkeypatch.setattr(preflight.sys, "platform", "win32")
    assert [preflight.gate_display_name(m)
            for m in preflight.N0_GATE_METRICS] == list(
                preflight.N0_GATE_METRICS)


def test_the_gate_display_name_never_moves_the_receipt_key(monkeypatch):
    """The negative control, and the reason this is a DISPLAY label.

    These strings are pre-registered N0 ledger record names read by
    gpuwm/verify/nest_gates.py and written into certification receipts.
    Renaming them per host would break every receipt written on one
    platform and read on another, so the tuple itself must not move.
    """
    from gpuwm.core import preflight
    from gpuwm.verify import nest_gates

    for platform in ("linux", "win32", "darwin", "freebsd13"):
        monkeypatch.setattr(preflight.sys, "platform", platform)
        assert preflight.N0_GATE_METRICS == (
            "alloc_fits_wddm_budget", "alloc_measured_le_estimate",
            "alloc_estimate_le_wddm_budget")
        # and the evaluated gate dict is still keyed by the record names
        gates = preflight.evaluate_alloc_gates(
            estimate_bytes=1, measured_used_bytes=1,
            measured_free_bytes=1 << 40,
            reserve=preflight.ReservePolicy(
                retention_residual_bytes=0, device_overhead_bytes=0))
        assert tuple(gates) == preflight.N0_GATE_METRICS

    # the verifier reads the keys, not the labels
    source = Path(nest_gates.__file__).read_text(encoding="utf-8")
    assert "alloc_fits_wddm_budget" in source
    assert "alloc_fits_vram_budget" not in source


# ---------------------------------------------------------------------------
# The subprocess device probe: the memory gate's only device access
# ---------------------------------------------------------------------------

class _CompletedProbe:
    """The slice of CompletedProcess the probe reads."""

    def __init__(self, returncode: int = 0, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


def test_the_device_probe_asks_everything_in_a_new_interpreter():
    """``memGetInfo``/``deviceGetLimit`` stand up a CUDA primary context
    wherever they are asked, so the probe must ask them in a NEW
    interpreter whose context dies with it -- a long-lived caller (the
    ``gpuwm go`` orchestrator, which outlives its gate as a progress
    printer) then holds no device memory at all."""
    import sys as _sys

    seen = {}

    def _runner(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        payload = {"free_bytes": 7 * GIB, "total_bytes": 8 * GIB,
                   "profile": {"name": "probe card",
                               "multiprocessor_count": 84,
                               "max_threads_per_multiprocessor": 1536,
                               "default_stack_limit_bytes": 1024}}
        return _CompletedProbe(stdout=json.dumps(payload) + "\n")

    payload = pf.device_memory_probe_subprocess(run=_runner)
    assert seen["command"][0] == _sys.executable
    assert seen["command"][1] == "-c"
    source = seen["command"][2]
    for question in ("memGetInfo", "getDeviceProperties", "deviceGetLimit"):
        assert question in source
    compile(source, "<device probe>", "exec")  # the source must parse
    assert (seen["kwargs"]["timeout"]
            == pf.DEVICE_MEMORY_PROBE_TIMEOUT_SECONDS)
    assert payload["free_bytes"] == 7 * GIB

    profile = pf.profile_from_device_probe(payload)
    assert profile is not None
    assert profile.name == "probe card"
    assert profile.multiprocessor_count == 84
    assert profile.max_threads_per_multiprocessor == 1536
    assert profile.default_stack_limit_bytes == 1024
    assert profile.resident_thread_capacity == 84 * 1536


def test_a_probe_that_cannot_answer_reads_as_no_device():
    """Every way the probe can fail is 'no card here', never a throw:
    the gate must never refuse on a card it could not see."""
    import subprocess

    for outcome in (
            _CompletedProbe(returncode=3),           # no cupy / no device
            _CompletedProbe(stdout="not json"),
            _CompletedProbe(stdout=""),
            _CompletedProbe(stdout=json.dumps({"free_bytes": "many"})),
            _CompletedProbe(stdout=json.dumps({"free_bytes": True})),
            _CompletedProbe(stdout=json.dumps(["free_bytes"]))):
        assert pf.device_memory_probe_subprocess(
            run=lambda *_a, _o=outcome, **_k: _o) is None

    def _timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 0))

    assert pf.device_memory_probe_subprocess(run=_timeout) is None

    def _unlaunchable(command, **kwargs):
        raise OSError("no interpreter")

    assert pf.device_memory_probe_subprocess(run=_unlaunchable) is None

    # ...and the profile half is as defensive as the free half.
    assert pf.profile_from_device_probe(None) is None
    assert pf.profile_from_device_probe({"free_bytes": 1}) is None
    assert pf.profile_from_device_probe({"profile": "a 5090"}) is None
    assert pf.profile_from_device_probe(
        {"profile": {"name": "half a card"}}) is None
