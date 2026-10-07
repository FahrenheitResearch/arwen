"""The radial-velocity dispersion gate.

CPU only.  THE BREAKAGE THIS PREVENTS: an under-dispersed radial-velocity
ensemble writes its innovations into theta and vapour through covariances
that are noise with a large gain; on a storm-scale first analysis that put
6.44 Mt of vapour into a box where no radar saw a storm.  The gate withholds
such a batch from theta and vapour in its under-dispersed columns only, and
only when the batch as a whole is under-dispersed
(:mod:`gpuwm.da.velocity_dispersion`).  These tests pin the rule, the
assembly, and the default route through ``assimilate_radar_grid``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core import constants as c
from gpuwm.da.letkf import GridGeometry, GriddedObs, Localization
from gpuwm.da.velocity_dispersion import (
    DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO, DEFAULT_VELOCITY_DISPERSION_RATIO,
    DispersionGate, DispersionGateError, velocity_dispersion, withhold)

NZ, NY, NX = 3, 12, 12
LOC = Localization(horizontal_m=6000.0, vertical_m=3000.0)
GEOMETRY = GridGeometry(dx_m=3000.0, dy_m=3000.0,
                        heights_m=np.array([500.0, 1500.0, 2500.0]))
LOCAL = {"geometry": GEOMETRY, "localization": LOC}


def _vr_batch(members, innovation, *, spread=0.5, error=1.0, seed=3,
              name="vr:TEST", column=(6, 6)):
    """A Vr batch observing every level of one column: ensemble H(x) with
    the stated spread about zero, observations ``innovation`` away."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((NZ, NY, NX), bool)
    mask[:, column[0], column[1]] = True
    simulated = np.zeros((members, NZ, NY, NX))
    draws = rng.standard_normal(members)
    draws = (draws - draws.mean()) / draws.std(ddof=1)
    simulated[:, mask] = spread * draws[:, None]
    values = np.where(mask, innovation, 0.0)
    return GriddedObs(name=name, values=values,
                      errors=np.full((NZ, NY, NX), error),
                      simulated=simulated, mask=mask, localization=LOC)


def test_the_defaults_are_the_measured_thresholds():
    assert DEFAULT_VELOCITY_DISPERSION_RATIO == 2.0
    assert DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO == 3.0


def test_an_under_dispersed_batch_is_gated_in_its_columns_only():
    batch = _vr_batch(6, 8.0)
    gates, receipt = velocity_dispersion(
        [batch], dx_m=3000.0, dy_m=3000.0, localization=LOC,
        shape=(NZ, NY, NX))
    entry = receipt["batches"][0]
    # 64 / (0.25 + 1)
    assert entry["batch_ratio"] == pytest.approx(51.2, rel=1e-9)
    assert entry["batch_gated"] is True
    assert len(gates) == 1
    columns = np.asarray(gates[0].columns)
    assert columns[6, 6]
    assert not columns[0, 0]
    assert gates[0].fields == ("thp", "qv")
    assert receipt["withheld_columns"] == int(columns.sum())


def test_a_batch_the_ensemble_explains_is_not_gated_but_is_recorded():
    # batch ratio 2.25 / 1.25 = 1.8: under the batch gate, every column of
    # it is above the column gate of 1.5 but none is withheld
    batch = _vr_batch(6, 1.5)
    gates, receipt = velocity_dispersion(
        [batch], dx_m=3000.0, dy_m=3000.0, localization=LOC,
        shape=(NZ, NY, NX), ratio=1.5)
    entry = receipt["batches"][0]
    assert entry["batch_gated"] is False
    assert entry["columns_above_gate"] > 0
    assert entry["withheld_columns"] == 0
    assert gates == []
    # the batch condition off: its columns alone gate it
    gates, _ = velocity_dispersion(
        [batch], dx_m=3000.0, dy_m=3000.0, localization=LOC,
        shape=(NZ, NY, NX), ratio=1.5, batch_ratio=None)
    assert len(gates) == 1


def test_the_gate_off_records_the_ratios_and_gates_nothing():
    batch = _vr_batch(6, 8.0)
    other = _vr_batch(6, 8.0, name="point:thp")
    gates, receipt = velocity_dispersion(
        [batch, other], dx_m=3000.0, dy_m=3000.0, localization=LOC,
        shape=(NZ, NY, NX), ratio=None)
    assert gates == []
    assert [e["batch"] for e in receipt["batches"]] == ["vr:TEST"]
    assert receipt["batches"][0]["batch_ratio"] > 3.0
    assert "batch_gated" not in receipt["batches"][0]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), "two"])
def test_a_ratio_the_rule_cannot_mean_is_refused(bad):
    with pytest.raises(DispersionGateError):
        velocity_dispersion([], dx_m=3000.0, dy_m=3000.0, localization=LOC,
                            shape=(NZ, NY, NX), ratio=bad)


def test_withhold_takes_the_gated_fields_from_the_solve_without_the_batch():
    members = 4
    rng = np.random.default_rng(7)
    fields = ("thp", "qv", "u")
    increments = {f: rng.standard_normal((members, NZ, NY, NX))
                  for f in fields}
    prior = {f: rng.standard_normal((members, NZ, NY, NX)) for f in fields}
    columns = np.zeros((NY, NX), bool)
    columns[5:8, 5:8] = True
    gate = DispersionGate(batch="vr:A", fields=("thp", "qv"), columns=columns)
    kept_batch = _vr_batch(members, 1.0, name="point:thp")
    calls = []

    def solve(gated_prior, batches, solved_fields, geometry):
        calls.append(([b.name for b in batches], solved_fields,
                      sorted(gated_prior)))
        shape = next(iter(gated_prior.values())).shape
        return {f: np.full(shape, 100.0 + i)
                for i, f in enumerate(solved_fields)}

    out, receipt = withhold(solve, prior,
                            [_vr_batch(members, 8.0, name="vr:A"), kept_batch],
                            increments, [gate], fields, **LOCAL)
    assert calls == [(["point:thp"], ("thp", "qv"), ["qv", "thp"])]
    for i, f in enumerate(("thp", "qv")):
        assert np.all(out[f][..., columns] == 100.0 + i)
        assert np.array_equal(out[f][..., ~columns],
                              increments[f][..., ~columns])
    assert np.array_equal(out["u"], increments["u"])
    # A field no gate rewrites is passed through, not copied: a second
    # whole-ensemble float64 copy of every analysed field was host memory
    # the gated solves never needed.  The rewritten ones are fresh arrays.
    assert out["u"] is increments["u"]
    assert out["thp"] is not increments["thp"]
    assert receipt["solves"][0]["columns"] == 9


def test_withhold_with_no_batch_left_solves_on_no_observation_over_the_zone():
    """No batch left anywhere: the zone still takes the solve, on an empty
    batch list over the box around the zone, and that solve's answer is
    written in the zone only."""
    members = 4
    increments = {"thp": np.ones((members, NZ, NY, NX))}
    columns = np.zeros((NY, NX), bool)
    columns[0, 0] = True
    columns[1, 2] = True
    gate = DispersionGate(batch="vr:A", fields=("thp", "qv"), columns=columns)
    calls = []

    def solve(gated_prior, batches, solved_fields, geometry):
        calls.append((list(batches), solved_fields,
                      {name: value.shape for name, value in
                       gated_prior.items()}))
        return {"thp": np.full(gated_prior["thp"].shape, 7.0)}

    out, receipt = withhold(solve, {"thp": increments["thp"]},
                            [_vr_batch(members, 8.0, name="vr:A")],
                            increments, [gate], ("thp",), **LOCAL)
    assert calls == [([], ("thp",), {"thp": (members, NZ, 2, 3)})]
    assert np.all(out["thp"][..., columns] == 7.0)
    assert np.all(out["thp"][..., ~columns] == 1.0)
    entry = receipt["solves"][0]
    assert entry["box"] == [0, 1, 0, 2]
    assert entry["batches_kept"] == 0 and entry["batches_out_of_reach"] == 0


def test_a_zone_no_batch_reaches_takes_the_solve_without_its_set_under_inflation():
    """Under prior inflation the filter's transform of a column no
    observation reaches is not zero (its perturbations grow by sqrt(rho)),
    so the withheld columns must take the solve without the withheld set
    whether batches are left elsewhere in the domain but none within
    reach, or no batch is left at all.  Both equal the whole-domain solve
    without the withheld set, and so equal each other."""
    from dataclasses import replace

    from gpuwm.da.letkf import LetkfConfig, analyze
    from gpuwm.da.radar_assimilation import letkf_grid_geometry

    members, ny, nx = 5, 28, 28
    grid = letkf_grid_geometry(_grid_sized(ny, nx))
    loc = Localization(horizontal_m=6000.0, vertical_m=3000.0)
    near = _column_batch(members, "vr:NEAR", (5, 5), ny=ny, nx=nx,
                         localization=loc, seed=1, radius=2)
    far = _column_batch(members, "z:FAR", (22, 22), ny=ny, nx=nx,
                        localization=loc, seed=2, radius=2)
    rng = np.random.default_rng(21)
    fields = ("thp", "qv", "u")
    prior = {f: rng.standard_normal((members, NZ, ny, nx)) for f in fields}
    config = LetkfConfig(localization=loc, analysis_fields=fields,
                         rtps_alpha=0.0, prior_inflation=1.2)
    columns = np.zeros((ny, nx), bool)
    columns[3:8, 3:8] = True
    gate = DispersionGate(batch="vr:NEAR", fields=("thp", "qv"),
                          columns=columns)

    def solve(gated_prior, kept, solved_fields, geometry):
        return analyze(gated_prior, kept, geometry,
                       replace(config, analysis_fields=tuple(solved_fields)))

    zone_values = []
    for batches, out_of_reach in (([near, far], 1), ([near], 0)):
        joint = analyze(prior, batches, grid, config)
        out, receipt = withhold(solve, prior, batches, joint, [gate], fields,
                                geometry=grid, localization=loc)
        whole = analyze(prior, [b for b in batches if b.name != "vr:NEAR"],
                        grid, replace(config, analysis_fields=("thp", "qv")))
        for f in ("thp", "qv"):
            # the inflation is there to lose: far from zero in the zone
            assert float(np.abs(whole[f][..., columns]).max()) > 1e-2
            np.testing.assert_allclose(out[f][..., columns],
                                       whole[f][..., columns],
                                       rtol=0, atol=1e-12)
            np.testing.assert_array_equal(out[f][..., ~columns],
                                          joint[f][..., ~columns])
        np.testing.assert_array_equal(out["u"], joint["u"])
        (entry,) = receipt["solves"]
        assert entry["batches_kept"] == 0
        assert entry["batches_out_of_reach"] == out_of_reach
        assert entry["box"] is not None
        zone_values.append({f: out[f][..., columns] for f in ("thp", "qv")})
    for f in ("thp", "qv"):
        np.testing.assert_allclose(zone_values[0][f], zone_values[1][f],
                                   rtol=0, atol=1e-12)


def test_withhold_without_a_gate_solves_nothing_and_returns_the_increments():
    increments = {"thp": np.ones((2, NZ, NY, NX))}
    out, receipt = withhold(lambda *_: pytest.fail("solved"), {}, [],
                            increments, [], ("thp",), **LOCAL)
    assert out is increments
    assert receipt["solves"] == []


def _column_batch(members, name, column, *, nz=NZ, ny=NY, nx=NX,
                  localization=LOC, window=None, seed=0, radius=0):
    """A batch observing every level of the columns within ``radius``
    cells of ``column``; with ``window`` its arrays cover that box only."""
    rng = np.random.default_rng(seed)
    jj, ii = np.mgrid[0:ny, 0:nx]
    full = np.zeros((nz, ny, nx), bool)
    full[:, np.hypot(jj - column[0], ii - column[1]) <= radius] = True
    if window is not None:
        j0, j1, i0, i1 = window
        full = full[:, j0:j1 + 1, i0:i1 + 1]
    simulated = rng.standard_normal((members,) + full.shape)
    values = np.where(full, 2.0 * rng.standard_normal(full.shape), 0.0)
    return GriddedObs(name=name, values=values,
                      errors=np.full(full.shape, 1.0), simulated=simulated,
                      mask=full, localization=localization, window=window)


def test_more_than_62_gated_batches_match_a_brute_force_answer():
    """66 gated batches, the width the int64 column code refused: every
    column's gated fields are the solve without exactly the batches gated
    there, found column by column."""
    import hashlib

    members, ny, nx = 3, 20, 20
    everywhere = Localization(horizontal_m=1.0e7, vertical_m=3000.0)
    geometry = GridGeometry(dx_m=3000.0, dy_m=3000.0,
                            heights_m=np.array([500.0, 1500.0]))
    rng = np.random.default_rng(62)
    batches = [_column_batch(members, f"vr:{k:03d}",
                             (int(rng.integers(ny)), int(rng.integers(nx))),
                             nz=2, ny=ny, nx=nx, localization=everywhere)
               for k in range(70)]
    gates = [DispersionGate(batch=f"vr:{k:03d}", fields=("thp", "qv"),
                            columns=rng.random((ny, nx)) < 0.15)
             for k in range(66)]
    names = {batch.name for batch in batches}
    fields = ("thp", "qv", "u")
    increments = {f: rng.standard_normal((members, 2, ny, nx))
                  for f in fields}

    def value(kept):
        digest = hashlib.sha256(",".join(sorted(kept)).encode()).hexdigest()
        return float(int(digest[:12], 16))

    def solve(_prior, kept, solved_fields, _geometry):
        worth = value(batch.name for batch in kept)
        shape = next(iter(_prior.values())).shape
        return {f: np.full(shape, worth + i)
                for i, f in enumerate(solved_fields)}

    out, receipt = withhold(solve, increments, batches, increments, gates,
                            fields, geometry=geometry,
                            localization=everywhere)
    sets = set()
    for j in range(ny):
        for i in range(nx):
            held = {gate.batch for gate in gates if gate.columns[j, i]}
            for n, f in enumerate(("thp", "qv")):
                if not held:
                    expected = increments[f][:, :, j, i]
                else:
                    expected = np.full((members, 2), value(names - held) + n)
                np.testing.assert_array_equal(out[f][:, :, j, i], expected)
            if held:
                sets.add(frozenset(held))
    np.testing.assert_array_equal(out["u"], increments["u"])
    assert len(receipt["solves"]) == len(sets)
    assert sum(entry["columns"] for entry in receipt["solves"]) == int(
        np.count_nonzero(np.any([gate.columns for gate in gates], axis=0)))


def test_the_zone_code_is_packbits_along_the_gate_axis():
    from gpuwm.da.velocity_dispersion import _zones

    rng = np.random.default_rng(5)
    stack = rng.random((70, 9, 11)) < 0.3
    gates = [DispersionGate(batch=f"vr:{k}", fields=("thp",),
                            columns=stack[k]) for k in range(70)]
    codes, inverse, gated = _zones(gates, 9, 11)
    packed = np.packbits(stack.reshape(70, -1), axis=0)
    np.testing.assert_array_equal(gated,
                                  np.flatnonzero(stack.any(axis=0).ravel()))
    np.testing.assert_array_equal(codes[:, inverse], packed[:, gated])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_reach_is_the_dilation_by_the_filters_stencil(seed):
    from gpuwm.da.letkf import _horizontal_stencil
    from gpuwm.da.velocity_dispersion import _dilate

    rng = np.random.default_rng(seed)
    where = rng.random((30, 40)) < 0.02
    spec = Localization(horizontal_m=float(rng.uniform(4000, 20000)),
                        vertical_m=3000.0)
    dj, di = _horizontal_stencil(spec, GEOMETRY, 40, 30)
    expected = np.zeros_like(where)
    for j, i in zip(*np.nonzero(where)):
        for a, b in zip(dj.tolist(), di.tolist()):
            if 0 <= j + a < 30 and 0 <= i + b < 40:
                expected[j + a, i + b] = True
    np.testing.assert_array_equal(_dilate(where, dj, di), expected)


def test_the_local_withheld_solve_is_the_whole_domain_solve_in_its_zone():
    """The real filter on the radar lane's geometry (heights per column,
    geodesic distances), several radars (two windowed), gates overlapping:
    each withheld solve runs on batches cut to its zone's reach, over the
    box around that reach only, and in the zone its increments equal the
    whole-domain solve without the same batches.  Far radars leave the
    local solve altogether."""
    from dataclasses import replace

    from gpuwm.da.letkf import LetkfConfig, analyze
    from gpuwm.da.radar_assimilation import letkf_grid_geometry

    members, nz, ny, nx = 5, NZ, 30, 30
    grid = letkf_grid_geometry(_grid_sized(ny, nx))
    loc = Localization(horizontal_m=9000.0, vertical_m=3000.0)
    sites = [(4, 4), (4, 25), (15, 15), (25, 5), (25, 25), (15, 3)]
    windows = {2: (10, 20, 10, 20), 4: (20, 29, 20, 29)}
    batches = [_column_batch(members, f"vr:R{k}", site, nz=nz, ny=ny, nx=nx,
                             localization=loc, window=windows.get(k), seed=k,
                             radius=4)
               for k, site in enumerate(sites)]
    batches.append(_column_batch(members, "z:R2", (15, 15), nz=nz, ny=ny,
                                 nx=nx, localization=loc, seed=9, radius=6))
    rng = np.random.default_rng(11)
    prior = {f: rng.standard_normal((members, nz, ny, nx))
             for f in ("thp", "qv", "u")}
    config = LetkfConfig(localization=loc, analysis_fields=("thp", "qv", "u"),
                         rtps_alpha=0.0)
    joint = analyze(prior, batches, grid, config)
    gates = []
    for k, (j, i) in enumerate(sites[:4]):
        columns = np.zeros((ny, nx), bool)
        columns[max(0, j - 5):j + 6, max(0, i - 5):i + 6] = True
        gates.append(DispersionGate(batch=f"vr:R{k}", fields=("thp", "qv"),
                                    columns=columns))
    seen, boxes = [], []

    def solve(gated_prior, kept, fields, geometry):
        seen.append(len(kept))
        boxes.append(next(iter(gated_prior.values())).shape[-2:])
        assert np.shape(geometry.lat_deg) == boxes[-1]
        return analyze(gated_prior, kept, geometry,
                       replace(config, analysis_fields=tuple(fields)))

    out, receipt = withhold(solve, prior, batches, joint, gates,
                            ("thp", "qv", "u"), geometry=grid,
                            localization=loc)
    assert receipt["solves"]
    assert any(entry["batches_out_of_reach"] for entry in receipt["solves"])
    assert max(seen) < len(batches)
    # the solves run on boxes smaller than the grid
    assert min(a * b for a, b in boxes) < ny * nx
    for entry in receipt["solves"]:
        held = set(entry["withheld"])
        zone = np.ones((ny, nx), bool)
        for gate in gates:
            zone &= gate.columns == (gate.batch in held)
        assert int(zone.sum()) == entry["columns"]
        whole = analyze(prior, [b for b in batches if b.name not in held],
                        grid, replace(config, analysis_fields=("thp", "qv")))
        for f in ("thp", "qv"):
            # the same analysis to rounding: the local solve's per-point
            # observation arrays are narrower, so a BLAS may sum them in
            # another order (bit for bit on some hosts, a few ulp on others)
            np.testing.assert_allclose(out[f][..., zone],
                                       whole[f][..., zone],
                                       rtol=0, atol=1e-12)
    np.testing.assert_array_equal(out["u"], joint["u"])


def _checkpoints(tmp_path, members):
    rng = np.random.default_rng(20260925)
    p = np.broadcast_to(np.array([95000.0, 90000.0, 85000.0])[:, None, None],
                        (NZ, NY, NX))
    t = np.broadcast_to(np.array([297.0, 294.0, 291.0])[:, None, None],
                        (NZ, NY, NX))
    paths = {}
    for index in range(members):
        qv = 0.006 + 0.0005 * rng.standard_normal((NZ, NY, NX))
        alt = c.RD * t * (1.0 + c.RVOVRD * qv) / p
        thp = 0.5 * rng.standard_normal((NZ, NY, NX))
        qc = np.abs(1.0e-4 * rng.standard_normal((NZ, NY, NX)))
        member_dir = tmp_path / f"member_{index:03d}"
        member_dir.mkdir()
        path = member_dir / "gpuwmrst_d01_000600.npz"
        np.savez(path, **{f"state/{k}": np.asarray(v, np.float32) for k, v in
                          {"qv": qv, "p": p, "alt": alt, "thp": thp,
                           "qc": qc}.items()})
        paths[index] = path
    return paths


def _grid():
    return _grid_sized(NY, NX)


def _grid_sized(ny, nx):
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid

    return TargetGrid.from_projection(
        LambertGrid(ref_lat=35.0, ref_lon=-97.0, truelat1=33.0, truelat2=37.0,
                    stand_lon=-97.0, dx=3000.0, dy=3000.0, e_we=nx + 1,
                    e_sn=ny + 1),
        z_w=np.linspace(0.0, 3000.0, NZ + 1), name="dispersion-gate-test")


def _run(paths, batches, **overrides):
    from gpuwm.da.radar_assimilation import (RadarAssimilationConfig,
                                             assimilate_radar_grid)

    cfg = RadarAssimilationConfig(
        solve_device="host", velocity=False, reflectivity=False,
        localization=LOC, rtps_alpha=0.0,
        analysis_fields=("thp", "qv", "qc"), positivity_policy="none",
        **overrides)
    return assimilate_radar_grid(
        paths, None, _grid(), cfg, extra_obs=batches,
        extra_obs_provenance=[{"source": "fixture"}])


def test_the_analysis_gates_by_default_and_matches_the_solve_without_vr(tmp_path):
    """End to end through assimilate_radar_grid, default configuration: in
    the gated columns theta and vapour are exactly the analysis without the
    Vr batch, the other field keeps the joint solve, and the receipt says
    so.  Switched off, Vr moves theta there."""
    members = 6
    paths = _checkpoints(tmp_path, members)
    vr = _vr_batch(members, 8.0, name="vr:TEST")
    point = _vr_batch(members, 1.0, name="point:thp", column=(2, 2), seed=5)
    gated, provenance = _run(paths, [vr, point])
    without_vr, _ = _run(paths, [point])
    joint, joint_provenance = _run(paths, [vr, point],
                                   velocity_dispersion_ratio=None)
    receipt = provenance["velocity_dispersion"]
    assert receipt["ratio"] == 2.0 and receipt["batch_ratio_gate"] == 3.0
    entry = receipt["batches"][0]
    assert entry["batch"] == "vr:TEST" and entry["batch_gated"] is True
    columns = np.zeros((NY, NX), bool)
    columns[6, 6] = True
    assert receipt["withheld"]["solves"][0]["columns"] >= 1
    for index in range(members):
        for f in ("thp", "qv"):
            a = np.asarray(gated[index][f])[..., 6, 6]
            b = np.asarray(without_vr[index][f])[..., 6, 6]
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(gated[index]["qc"]),
                                   np.asarray(joint[index]["qc"]),
                                   rtol=0, atol=0)
    moved = max(float(np.abs(np.asarray(joint[i]["thp"])[..., 6, 6]
                             - np.asarray(without_vr[i]["thp"])[..., 6, 6]).max())
                for i in range(members))
    assert moved > 1e-3
    assert joint_provenance["velocity_dispersion"]["ratio"] is None
    assert joint_provenance["velocity_dispersion"]["withheld"]["solves"] == []


def test_the_config_refuses_a_ratio_the_rule_cannot_mean():
    from gpuwm.da.radar_assimilation import (RadarAssimilationConfig,
                                             RadarAssimilationError)

    with pytest.raises(RadarAssimilationError, match="velocity_dispersion"):
        RadarAssimilationConfig(localization=LOC, rtps_alpha=0.0,
                                velocity_dispersion_ratio=0.0)


def test_the_door_passes_the_gate_and_takes_none():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "dcp_gate_probe",
        Path(__file__).resolve().parent.parent / "tools"
        / "da_cycle_prepared.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._dispersion_ratio_argument("none") is None
    assert module._dispersion_ratio_argument("2.5") == 2.5
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        module._dispersion_ratio_argument("0")


def test_the_door_prints_the_gate_every_run():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "dcp_gate_line_probe",
        Path(__file__).resolve().parent.parent / "tools"
        / "da_cycle_prepared.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    line = module.dispersion_gate_line
    assert line(2.0, 3.0) == "velocity dispersion gate: column 2, batch 3 (default)"
    assert line(2.5, 3.0) == "velocity dispersion gate: column 2.5, batch 3 (set)"
    assert line(2.0, None) == ("velocity dispersion gate: column 2, batch "
                               "none, every batch gated on its columns "
                               "alone (set)")
    assert line(None, 3.0).startswith("velocity dispersion gate: off")
    source = (Path(__file__).resolve().parent.parent / "tools"
              / "da_cycle_prepared.py").read_text(encoding="utf-8")
    # printed unconditionally, beside the report entries it describes
    assert "    print(dispersion_gate_line(" in source
    assert "if args.velocity_dispersion_gate is None:" not in source


def test_an_ab_bundle_replays_the_gate_it_recorded():
    import importlib.util
    from pathlib import Path

    from gpuwm.da.radar_assimilation import RadarAssimilationConfig

    spec = importlib.util.spec_from_file_location(
        "solve_ab_gate_probe",
        Path(__file__).resolve().parent.parent / "tools" / "da_solve_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = RadarAssimilationConfig(localization=LOC, rtps_alpha=0.0)
    recorded = module.serialize_config(base)
    assert recorded["velocity_dispersion_ratio"] == 2.0
    replay = module.build_config({"config": recorded}, "host")
    assert replay.velocity_dispersion_ratio == 2.0
    assert replay.velocity_dispersion_batch_ratio == 3.0
    # recorded before the gate existed, or with it off: replays without it
    for name in ("velocity_dispersion_ratio",
                 "velocity_dispersion_batch_ratio"):
        recorded.pop(name)
    replay = module.build_config({"config": recorded}, "host")
    assert replay.velocity_dispersion_ratio is None
    assert replay.velocity_dispersion_batch_ratio is None


def test_a_point_batch_is_reindexed_into_each_withheld_solve_box():
    """A conventional point batch (whole-grid flat indices) beside gated Vr:
    each zone's cut solve must receive it re-indexed into the box with no
    window -- the device solver refuses a point batch with a window -- and
    holding only the points the box keeps; in the zone the solve equals the
    whole-domain solve without the withheld batches.  The Iowa da-obs-iau
    cycle (2026-10-06) stopped on exactly this at its first analysis."""
    from dataclasses import replace

    from gpuwm.da.letkf import LetkfConfig, PointSet, analyze, point_batch
    from gpuwm.da.radar_assimilation import letkf_grid_geometry

    members, nz, ny, nx = 5, NZ, 30, 30
    grid = letkf_grid_geometry(_grid_sized(ny, nx))
    loc = Localization(horizontal_m=9000.0, vertical_m=3000.0)
    sites = [(4, 4), (25, 25), (15, 15)]
    batches = [_column_batch(members, f"vr:R{k}", site, nz=nz, ny=ny, nx=nx,
                             localization=loc, seed=k, radius=4)
               for k, site in enumerate(sites)]
    rng = np.random.default_rng(7)
    cells = [(0, 3, 6), (1, 5, 5), (2, 14, 16), (0, 26, 24), (1, 28, 2)]
    flat = np.sort(np.ravel_multi_index(tuple(np.array(cells).T), (nz, ny, nx)))
    points = PointSet(flat_index=flat, values=rng.standard_normal(flat.size),
                      errors=np.full(flat.size, 1.0),
                      simulated=rng.standard_normal((members, flat.size)))
    batches.append(point_batch("conventional:surface-temperature",
                               (nz, ny, nx), points, localization=loc))
    prior = {f: rng.standard_normal((members, nz, ny, nx))
             for f in ("thp", "qv", "u")}
    config = LetkfConfig(localization=loc, analysis_fields=("thp", "qv", "u"),
                         rtps_alpha=0.0)
    joint = analyze(prior, batches, grid, config)
    gates = []
    for k, (j, i) in enumerate(sites[:2]):
        columns = np.zeros((ny, nx), bool)
        columns[max(0, j - 5):j + 6, max(0, i - 5):i + 6] = True
        gates.append(DispersionGate(batch=f"vr:R{k}", fields=("thp", "qv"),
                                    columns=columns))
    seen_points = []

    def solve(gated_prior, kept, fields, geometry):
        shape = next(iter(gated_prior.values())).shape[1:]
        for b in kept:
            if b.points is None:
                continue
            assert b.window is None
            idx = np.asarray(b.points.flat_index)
            assert idx.size == int(np.count_nonzero(b.mask))
            assert idx.size == 0 or int(idx.max()) < int(np.prod(shape))
            assert np.shape(b.mask) == tuple(shape)
            seen_points.append(int(idx.size))
        return analyze(gated_prior, kept, geometry,
                       replace(config, analysis_fields=tuple(fields)))

    out, receipt = withhold(solve, prior, batches, joint, gates,
                            ("thp", "qv", "u"), geometry=grid,
                            localization=loc)
    assert receipt["solves"] and seen_points
    assert min(seen_points) < flat.size          # some box dropped points
    for entry in receipt["solves"]:
        held = set(entry["withheld"])
        zone = np.ones((ny, nx), bool)
        for gate in gates:
            zone &= gate.columns == (gate.batch in held)
        whole = analyze(prior, [b for b in batches if b.name not in held],
                        grid, replace(config, analysis_fields=("thp", "qv")))
        for f in ("thp", "qv"):
            np.testing.assert_allclose(out[f][..., zone], whole[f][..., zone],
                                       rtol=0, atol=1e-12)
