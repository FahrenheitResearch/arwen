"""The device LETKF's in-filter stages and its cards, against the routes
they replace, byte for byte.

THE BREAKAGE THIS PREVENTS: on the 9 km CONUS case (598 x 351 x 49, 32
members) the analysis spent 114 s re-solving theta and vapour in 391 small
cropped solves for the radial-velocity dispersion gate, 35 to 265 s on one
host core enforcing positivity, and ran all 51 filter chunks on one card of
eight.  The device route now re-solves the gated columns and applies the
positivity policy inside its own pass, chunk by chunk on whichever card
took the chunk.  None of that may change a byte of the analysis: these
tests compare each new path with the old one with ``tobytes``.
"""
from __future__ import annotations

from dataclasses import replace

import cupy as cp
import numpy as np
import pytest

from gpuwm.da import letkf_device
from gpuwm.da.letkf import LetkfConfig, LetkfDiagnostics, Localization
from gpuwm.da.letkf_device import ColumnWithhold, analyze_device
from gpuwm.da.positivity import (DevicePositivity, PositivityError,
                                 apply_positivity, verify_non_negative)
from gpuwm.da.velocity_dispersion import (DispersionGate, column_plan,
                                          withhold)
from test_letkf_device_gpu import geodesic_problem
from test_velocity_dispersion_gate import NZ, _column_batch, _grid_sized

pytestmark = pytest.mark.gpu


def _device(prior, obs, grid, cfg, **kw):
    diag = LetkfDiagnostics()
    return analyze_device(prior, obs, grid, cfg, diag, **kw), diag


def _same(a, b, names):
    for name in names:
        assert a[name].dtype == b[name].dtype, name
        assert a[name].tobytes() == b[name].tobytes(), name


# -- several cards (two workers on one card here; real cards on a box) -------

def test_two_workers_give_one_cards_bytes_and_diagnostics(monkeypatch):
    monkeypatch.setattr(letkf_device, "DIAG_BLOCK", 64)
    prior, obs, grid, cfg = geodesic_problem(seed=7)
    cfg = replace(cfg, chunk_points=None)
    one, d1 = _device(prior, obs, grid, cfg, chunk_points=128)
    two, d2 = _device(prior, obs, grid, cfg, chunk_points=128,
                      devices=[0, 0])
    small, d3 = _device(prior, obs, grid, cfg, chunk_points=64,
                        devices=[0, 0, 0])
    assert d2.device_chunks > 2 and d3.device_chunks > d2.device_chunks
    _same(one, two, prior)
    _same(one, small, prior)
    # whole blocks per chunk: the receipt's means are the same bytes too
    for f in prior:
        assert d1.prior_spread[f] == d2.prior_spread[f] == d3.prior_spread[f]
        assert (d1.posterior_spread[f] == d2.posterior_spread[f]
                == d3.posterior_spread[f])
        assert (d1.mean_increment_rms[f] == d2.mean_increment_rms[f]
                == d3.mean_increment_rms[f])
    assert d1.active_points == d2.active_points == d3.active_points
    assert d1.max_local_obs == d2.max_local_obs == d3.max_local_obs


def test_all_visible_cards_give_the_same_bytes():
    prior, obs, grid, cfg = geodesic_problem(seed=8)
    one, _ = _device(prior, obs, grid, cfg)
    every, diag = _device(prior, obs, grid, cfg, devices="all")
    _same(one, every, prior)
    assert diag.devices == list(range(cp.cuda.runtime.getDeviceCount()))


# -- the dispersion gate inside the pass ------------------------------------

def _gated_problem(members=8, ny=30, nx=30):
    from gpuwm.da.radar_assimilation import letkf_grid_geometry

    grid = letkf_grid_geometry(_grid_sized(ny, nx))
    loc = Localization(horizontal_m=9000.0, vertical_m=3000.0)
    sites = [(4, 4), (4, 25), (15, 15), (25, 5), (25, 25), (15, 3)]
    windows = {2: (10, 20, 10, 20), 4: (20, 29, 20, 29)}
    batches = [_column_batch(members, f"vr:R{k}", site, nz=NZ, ny=ny, nx=nx,
                             localization=loc, window=windows.get(k), seed=k,
                             radius=4)
               for k, site in enumerate(sites)]
    batches.append(_column_batch(members, "z:R2", (15, 15), nz=NZ, ny=ny,
                                 nx=nx, localization=loc, seed=9, radius=6))
    rng = np.random.default_rng(11)
    prior = {f: rng.standard_normal((members, NZ, ny, nx))
             for f in ("thp", "qv", "u")}
    config = LetkfConfig(localization=loc,
                         analysis_fields=("thp", "qv", "u"),
                         rtps_alpha=0.9)
    gates = []
    for k, (j, i) in enumerate(sites[:4]):
        columns = np.zeros((ny, nx), bool)
        columns[max(0, j - 5):j + 6, max(0, i - 5):i + 6] = True
        gates.append(DispersionGate(batch=f"vr:R{k}", fields=("thp", "qv"),
                                    columns=columns))
    return prior, batches, grid, config, gates, loc


@pytest.mark.parametrize("rho", [1.0, 1.2])
def test_in_filter_withhold_is_the_cropped_solves_byte_for_byte(rho):
    prior, batches, grid, config, gates, loc = _gated_problem()
    config = replace(config, prior_inflation=rho)
    joint, _ = _device(prior, batches, grid, config)

    def solve(gated_prior, kept, fields, geometry):
        out, _ = _device(gated_prior, kept, geometry,
                         replace(config, analysis_fields=tuple(fields)))
        return out

    old, receipt = withhold(solve, prior, batches, joint, gates,
                            ("thp", "qv", "u"), geometry=grid,
                            localization=loc)
    assert len(receipt["solves"]) > 1
    fields, column_zone, zones, zone_receipt = column_plan(
        gates, ("thp", "qv", "u"))
    assert fields == ("thp", "qv")
    plan = ColumnWithhold(fields=fields, column_zone=column_zone, zones=zones)
    new, diag = _device(prior, batches, grid, config, withhold=plan)
    _same(old, new, ("thp", "qv", "u"))
    assert diag.withheld["points"] == NZ * int((column_zone >= 0).sum())
    assert sorted((e["withheld"], e["columns"]) for e in zone_receipt) == \
        sorted((e["withheld"], e["columns"]) for e in receipt["solves"])
    # and with chunks cut across zones, on two workers
    split, _ = _device(prior, batches, grid, config, withhold=plan,
                       chunk_points=97, devices=[0, 0])
    _same(old, split, ("thp", "qv", "u"))


# -- positivity on the card --------------------------------------------------

def _positive_problem(seed=3):
    prior, obs, grid, cfg = geodesic_problem(seed=seed, fields=4)
    names = ("qv", "qc", "thp", "qr")
    prior = {new: np.abs(prior[old]) * (0.01 if new != "thp" else 1.0)
             for new, old in zip(names, sorted(prior))}
    # some zero background, where any negative increment must clip
    prior["qc"][:, :2] = 0.0
    cfg = replace(cfg, analysis_fields=names)
    return prior, obs, grid, cfg


@pytest.mark.parametrize("policy", ["mean-preserving", "clip", "reject", "none"])
def test_device_positivity_is_the_host_policy_byte_for_byte(policy):
    prior, obs, grid, cfg = _positive_problem()
    raw, _ = _device(prior, obs, grid, cfg)
    host, host_receipt = apply_positivity(prior, raw, policy=policy,
                                          fields=cfg.analysis_fields)
    if policy != "none":
        verify_non_negative(prior, host, fields=cfg.analysis_fields)
    assert host_receipt["negative_points"] > 0
    for chunk, devices in ((None, None), (53, [0, 0])):
        hook = DevicePositivity(policy, fields=cfg.analysis_fields)
        new, diag = _device(prior, obs, grid, cfg, chunk_hook=hook,
                            chunk_points=chunk, devices=devices)
        _same(host, new, cfg.analysis_fields)
        assert diag.chunk_hook_result == host_receipt


def test_device_positivity_refuses_what_verify_refuses():
    prior, obs, grid, cfg = _positive_problem()
    prior["qr"][0, 0, 0, 0] = -1.0     # a negative background
    hook = DevicePositivity("reject", fields=cfg.analysis_fields)
    with pytest.raises(PositivityError, match="still negative"):
        _device(prior, obs, grid, cfg, chunk_hook=hook)


# -- the prior as the members' own arrays -------------------------------------

def test_a_member_stack_prior_gives_the_stacked_priors_bytes():
    """assimilate_radar_grid hands the device route a MemberStack per field
    (the members' arrays, widened chunk by chunk) instead of a whole float64
    stack; the increments are the stacked prior's bytes, and np.asarray of
    the stack is that stack."""
    from gpuwm.da.radar_assimilation import MemberStack, _mass_field

    prior, obs, grid, cfg = geodesic_problem(seed=12, members=16)
    names = tuple(prior)
    states = {m: {name: prior[name][m].astype(np.float32) for name in names}
              for m in range(16)}
    shape = prior[names[0]].shape[1:]
    lazy = {name: MemberStack(name, states, list(range(16)), shape)
            for name in names}
    stacked = {name: np.stack([_mass_field(name, states[m], "m")
                               for m in range(16)]) for name in names}
    for name in names:
        assert np.asarray(lazy[name]).tobytes() == stacked[name].tobytes()
    lazy = {name: MemberStack(name, states, list(range(16)), shape)
            for name in names}
    a, _ = _device(stacked, obs, grid, cfg, chunk_points=53, devices=[0, 0])
    b, _ = _device(lazy, obs, grid, cfg, chunk_points=53, devices=[0, 0])
    _same(a, b, names)
    assert all(lazy[name]._whole is None for name in names)   # never stacked


@pytest.mark.parametrize("policy", ["mean-preserving", "clip"])
def test_device_positivity_on_a_float32_prior_is_the_host_policy(policy):
    """The filter returns float32 increments for a float32 prior; the host
    policy acts on those and casts back (mean-preserving floors a cast that
    lands a rounding below zero).  The hook rounds to the same dtype first
    and last, so the bytes and the receipt are the host's."""
    prior, obs, grid, cfg = _positive_problem(seed=5)
    prior = {k: v.astype(np.float32) for k, v in prior.items()}
    raw, _ = _device(prior, obs, grid, cfg)
    assert raw["qv"].dtype == np.float32
    host, host_receipt = apply_positivity(prior, raw, policy=policy,
                                          fields=cfg.analysis_fields)
    hook = DevicePositivity(policy, fields=cfg.analysis_fields)
    new, diag = _device(prior, obs, grid, cfg, chunk_hook=hook,
                        chunk_points=61, devices=[0, 0])
    _same(host, new, cfg.analysis_fields)
    assert diag.chunk_hook_result == host_receipt


def test_two_withheld_groups_are_the_sequential_host_withholds():
    """Field rules give several gate groups with disjoint field sets; the
    device re-solves every group in its one pass, byte for byte the host's
    sequential withhold of each group."""
    prior, batches, grid, config, gates, loc = _gated_problem()
    hydro = []
    for k, (j, i) in enumerate([(15, 15), (25, 25)]):
        columns = np.zeros(gates[0].columns.shape, bool)
        columns[j - 4:j + 5, i - 4:i + 5] = True
        hydro.append(DispersionGate(batch=["z:R2", "vr:R4"][k], fields=("u",),
                                    columns=columns))
    groups = [gates, hydro]
    joint, _ = _device(prior, batches, grid, config)

    def solve(gated_prior, kept, fields, geometry):
        out, _ = _device(gated_prior, kept, geometry,
                         replace(config, analysis_fields=tuple(fields)))
        return out

    old = joint
    for group in groups:
        old, _ = withhold(solve, prior, batches, old, group,
                          ("thp", "qv", "u"), geometry=grid, localization=loc)
    plans = [column_plan(group, ("thp", "qv", "u")) for group in groups]
    new, diag = _device(prior, batches, grid, config, chunk_points=101,
                        devices=[0, 0],
                        withhold=tuple(ColumnWithhold(fields=f, column_zone=c,
                                                      zones=z)
                                       for f, c, z, _r in plans))
    _same(old, new, ("thp", "qv", "u"))
    assert [g["fields"] for g in diag.withheld["groups"]] == [["thp", "qv"],
                                                             ["u"]]


def test_a_keep_weight_blends_the_joint_and_withheld_solves_as_the_host_does():
    """A gate with keep w gives keep*joint + (1-keep)*withheld in its
    columns; the in-pass device route equals the host withhold byte for
    byte, and keep 0 is the plain withhold."""
    prior, batches, grid, config, gates, loc = _gated_problem()
    gates = [DispersionGate(batch=g.batch, fields=g.fields, columns=g.columns,
                            keep=0.1) for g in gates]
    joint, _ = _device(prior, batches, grid, config)

    def solve(gated_prior, kept, fields, geometry):
        out, _ = _device(gated_prior, kept, geometry,
                         replace(config, analysis_fields=tuple(fields)))
        return out

    old, receipt = withhold(solve, prior, batches, joint, gates,
                            ("thp", "qv", "u"), geometry=grid, localization=loc)
    assert all(entry["keep"] == 0.1 for entry in receipt["solves"])
    f, c, z, r = column_plan(gates, ("thp", "qv", "u"))
    new, _ = _device(prior, batches, grid, config, chunk_points=97,
                     devices=[0, 0],
                     withhold=ColumnWithhold(fields=f, column_zone=c, zones=z,
                                             keep=tuple(e["keep"] for e in r)))
    _same(old, new, ("thp", "qv", "u"))
    plain, _ = _device(prior, batches, grid, config,
                       withhold=ColumnWithhold(fields=f, column_zone=c, zones=z))
    assert any(plain[k].tobytes() != new[k].tobytes() for k in ("thp", "qv"))


def test_gates_with_different_keeps_at_one_column_are_refused():
    from gpuwm.da.velocity_dispersion import DispersionGateError
    _prior, _b, _g, _c, gates, _loc = _gated_problem()
    mixed = [DispersionGate(batch=g.batch, fields=g.fields, columns=g.columns,
                            keep=0.1 * (k % 2)) for k, g in enumerate(gates)]
    with pytest.raises(DispersionGateError, match="different keeps"):
        column_plan(mixed, ("thp", "qv", "u"))
