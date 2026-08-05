"""The real-radar assimilate(): checkpoints in, seam-shaped increments out.

CPU only.  The observation file is written by the radar lane's own writer
(never a dict pretending to be a file), the checkpoints carry the restart
layout (``state/`` arrays, ARW staggering), and the end-to-end test hands
the callable exactly what :func:`gpuwm.ensemble.cycle.run_cycles` hands it.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from gpuwm.da import obsop
from gpuwm.da.letkf import Localization
from gpuwm.da.radar_assimilation import (
    FALL_SPEED_POLICIES, RadarAssimilationConfig, RadarAssimilationError,
    assimilate_radar_grid, grid_rotation, innovation_summary,
    make_assimilate, mass_to_u_faces, mass_to_v_faces, mass_to_w_faces,
    member_background_checkpoint, read_checkpoint_state,
    scheme_reflectivity_provider)

NZ, NY, NX = 6, 20, 20
DX_M = 3000.0
TOP_M = 12000.0
MEMBERS = 8
SEED = 20260731
#: Rows/columns of unobserved margin; wide enough that the far corner is
#: outside every observation's localisation lens.
MARGIN = 7
H_LOC_M = 15000.0
V_LOC_M = 4000.0
OBS_ERR_MS = 1.0
#: The analysed set once moisture and hydrometeors are in it.  Spelled
#: here rather than at each call site so a test that adds a species adds
#: it everywhere the invariants are checked.
HYDRO_FIELDS = ("u", "v", "thp", "qv", "qr")


# ---------------------------------------------------------------------------
# world building
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid():
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid

    projection = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.0, dx=DX_M, dy=DX_M, e_we=NX + 1, e_sn=NY + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, TOP_M, NZ + 1),
        name="radar-assim-test")


def smooth(field):
    """Two-pass three-point average: a cheap correlated field."""
    for _ in range(2):
        for axis in (1, 2):
            field = (np.roll(field, 1, axis) + field
                     + np.roll(field, -1, axis)) / 3.0
    return field


def correlated_noise(rng, sigma, shape):
    """Spatially correlated noise with the requested pointwise sigma.

    An ensemble perturbed with WHITE noise defeats an 8-member filter --
    sampled covariances between a gridpoint and its neighbours are pure
    sampling error -- and real IC perturbations are correlated by
    construction (gpuwm.da.perturb draws from a spectrum).  This is the
    cheap stand-in for that property.
    """
    field = smooth(rng.normal(0.0, 1.0, shape))
    return sigma * field / field.std()


def _mass_truth(rng):
    """Smooth-ish mass-point truth winds and moisture."""
    truth = {
        "u": 8.0 + smooth(rng.normal(0.0, 3.0, (NZ, NY, NX))),
        "v": -2.0 + smooth(rng.normal(0.0, 3.0, (NZ, NY, NX))),
        "w": smooth(rng.normal(0.0, 0.5, (NZ, NY, NX))),
        "thp": smooth(rng.normal(0.0, 1.0, (NZ, NY, NX))),
        "qv": np.full((NZ, NY, NX), 8.0e-3),
        "qr": np.maximum(smooth(rng.normal(5.0e-4, 5.0e-4, (NZ, NY, NX))),
                         0.0),
    }
    return truth


def _stagger(name, mass):
    if name == "u":
        return mass_to_u_faces(mass)
    if name == "v":
        return mass_to_v_faces(mass)
    if name == "w":
        return mass_to_w_faces(mass)
    return mass


def _write_checkpoint(path: Path, fields: dict) -> None:
    payload = {f"state/{name}": np.asarray(values, np.float32)
               for name, values in fields.items()}
    np.savez(path, **payload)


def _member_fields(center, rng):
    """One member: an independent draw around the ensemble centre, in
    checkpoint (staggered) layout, with full pressure.

    Truth is its OWN draw around the same centre (the twin-experiment
    construction :mod:`gpuwm.da.synthetic_cycle` uses), so the prior
    mean error and the ensemble spread match by construction -- a
    calibrated toy, which is the setting in which "the analysis must
    beat the prior" is a fair demand of an 8-member filter.
    """
    fields = {}
    for name, sigma in (("u", 1.5), ("v", 1.5), ("w", 0.3)):
        mass = center[name] + correlated_noise(rng, sigma, (NZ, NY, NX))
        fields[name] = _stagger(name, mass)
    fields["thp"] = center["thp"] + correlated_noise(rng, 0.5, (NZ, NY, NX))
    fields["qv"] = center["qv"]
    fields["qr"] = np.maximum(
        center["qr"] + rng.normal(0.0, 2.0e-4, (NZ, NY, NX)), 0.0)
    column = np.linspace(9.0e4, 4.0e4, NZ)
    fields["p"] = np.broadcast_to(column[:, None, None],
                                  (NZ, NY, NX)).copy()
    return fields


@pytest.fixture(scope="module")
def world(grid, tmp_path_factory):
    """Truth, an ensemble of member directories, and a real obs file."""
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.superob import GriddedObservations, SuperobParams

    root = tmp_path_factory.mktemp("radar-assim")
    rng = np.random.default_rng(SEED)
    center = _mass_truth(rng)
    # Truth: its own draw around the centre, not one of the R members.
    truth = {name: values.copy() for name, values in center.items()}
    for name, sigma in (("u", 1.5), ("v", 1.5), ("w", 0.3),
                        ("thp", 0.5)):
        truth[name] = center[name] + correlated_noise(
            rng, sigma, (NZ, NY, NX))

    # -- members: real checkpoint layout in real member directories --------
    leg_root = root / "cycle_000"
    member_states = {}
    for index in range(MEMBERS):
        member_dir = leg_root / f"member_{index:03d}"
        member_dir.mkdir(parents=True)
        _write_checkpoint(member_dir / "gpuwmrst_d01_000600.npz",
                          _member_fields(center, rng))
        member_states[index] = {"member_dir": str(member_dir)}

    # -- observations through the radar lane's own writer ------------------
    sites = [
        obsop.RadarSite(latitude_deg=float(grid.lat[NY // 2, 1]),
                        longitude_deg=float(grid.lon[NY // 2, 1]),
                        altitude_m=350.0, name="AAAA"),
        obsop.RadarSite(latitude_deg=float(grid.lat[1, NX // 2]),
                        longitude_deg=float(grid.lon[1, NX // 2]),
                        altitude_m=380.0, name="BBBB"),
    ]
    geometry = obsop.GridGeometry.from_target_grid(grid)
    shape = (NZ, NY, NX)
    interior = np.zeros(shape, bool)
    interior[:, MARGIN:NY - MARGIN, MARGIN:NX - MARGIN] = True

    sina, cosa = grid_rotation(grid)
    u_e, v_n = obsop.earth_relative_winds(truth["u"], truth["v"],
                                          sina, cosa)

    east = np.zeros((2,) + shape)
    north = np.zeros((2,) + shape)
    up = np.zeros((2,) + shape)
    vr_obs = np.zeros((2,) + shape)
    for slot, site in enumerate(sites):
        beam = obsop.beam_geometry(geometry, site)
        e, n, u = (np.broadcast_to(np.asarray(c, np.float64), shape).copy()
                   for c in beam.unit_vector_enu())
        east[slot], north[slot], up[slot] = e, n, u
        clean = u_e * e + v_n * n + truth["w"] * u
        vr_obs[slot] = clean + rng.normal(0.0, OBS_ERR_MS, shape)

    vr_mask = np.broadcast_to(interior, (2,) + shape).astype(np.int8)
    zeros = np.zeros(shape, np.float32)
    observations = GriddedObservations(
        z_obs=zeros, z_mask=np.zeros(shape, np.int8), z_err=zeros,
        z_max=zeros, z_mean=zeros, z_count=np.zeros(shape, np.int32),
        vr_obs=vr_obs.astype(np.float32), vr_mask=vr_mask,
        vr_err=np.where(vr_mask.astype(bool), OBS_ERR_MS,
                        0.0).astype(np.float32),
        vr_count=vr_mask.astype(np.int32),
        vr_rejected=np.zeros((2,) + shape, np.int32),
        vr_beam_east=east.astype(np.float32),
        vr_beam_north=north.astype(np.float32),
        vr_beam_up=up.astype(np.float32),
        radars=[{"id": site.name, "lat_deg": site.latitude_deg,
                 "lon_deg": site.longitude_deg, "alt_m": site.altitude_m,
                 "valid_time": "2026-01-01T00:00:00Z"} for site in sites],
        counts=[], provenance=[])
    obs_path = root / "obs-radar-grid.nc"
    write_radar_grid(obs_path, observations, grid,
                     valid_time="2026-01-01T00:00:00Z",
                     params=SuperobParams(), overwrite=True)

    return types.SimpleNamespace(
        root=root, truth=truth, member_states=member_states,
        obs_path=obs_path, interior=interior)


def _config(**overrides):
    kwargs = dict(localization=Localization(horizontal_m=H_LOC_M,
                                            vertical_m=V_LOC_M),
                  rtps_alpha=0.9, analysis_fields=("u", "v"))
    kwargs.update(overrides)
    return RadarAssimilationConfig(**kwargs)


# ---------------------------------------------------------------------------
# staggering round trips
# ---------------------------------------------------------------------------


def test_restagger_shapes_and_constant_preservation():
    mass = np.full((NZ, NY, NX), 3.25)
    u = mass_to_u_faces(mass)
    v = mass_to_v_faces(mass)
    w = mass_to_w_faces(mass)
    assert u.shape == (NZ, NY, NX + 1)
    assert v.shape == (NZ, NY + 1, NX)
    assert w.shape == (NZ + 1, NY, NX)
    for faces in (u, v, w):
        assert np.all(faces == 3.25)
    # Constant fields survive the full round trip bit-exactly.
    assert np.array_equal(obsop.destagger_u(u), mass)
    assert np.array_equal(obsop.destagger_v(v), mass)
    assert np.array_equal(obsop.destagger_w(w), mass)


def test_restagger_interior_is_adjacent_mean():
    rng = np.random.default_rng(7)
    mass = rng.normal(size=(NZ, NY, NX))
    u = mass_to_u_faces(mass)
    assert np.allclose(u[:, :, 1:NX],
                       0.5 * (mass[:, :, :-1] + mass[:, :, 1:]))
    assert np.array_equal(u[:, :, 0], mass[:, :, 0])
    assert np.array_equal(u[:, :, NX], mass[:, :, -1])


def test_restagger_refuses_wrong_rank():
    with pytest.raises(RadarAssimilationError):
        mass_to_u_faces(np.zeros((NY, NX)))


# ---------------------------------------------------------------------------
# checkpoint discovery binds to the driver's own rule
# ---------------------------------------------------------------------------


def test_member_background_checkpoint_matches_cycle_driver(tmp_path):
    from gpuwm.ensemble.cycle import _member_background_checkpoint

    member = tmp_path / "member_000"
    member.mkdir()
    for name in ("gpuwmrst_d01_000300.npz", "gpuwmrst_d01_000600.npz",
                 "gpuwmrst_d01_000150.npz"):
        np.savez(member / name, **{"state/u": np.zeros(1, np.float32)})
    ours = member_background_checkpoint(member)
    theirs = _member_background_checkpoint(member)
    assert ours == theirs
    assert ours.name == "gpuwmrst_d01_000600.npz"


def test_member_background_checkpoint_fails_closed(tmp_path):
    empty = tmp_path / "member_001"
    empty.mkdir()
    with pytest.raises(RadarAssimilationError, match="no gpuwmrst"):
        member_background_checkpoint(empty)


def test_read_checkpoint_state_refuses_missing_field(tmp_path):
    path = tmp_path / "gpuwmrst_d01_000600.npz"
    np.savez(path, **{"state/u": np.zeros((2, 2, 3), np.float32)})
    with pytest.raises(RadarAssimilationError, match="does not carry"):
        read_checkpoint_state(path, fields=["u", "v"])


# ---------------------------------------------------------------------------
# configuration refusals
# ---------------------------------------------------------------------------


def test_config_refuses_assimilating_nothing():
    with pytest.raises(RadarAssimilationError, match="assimilate nothing"):
        _config(velocity=False, reflectivity=False)


def test_config_refuses_unknown_z_source():
    with pytest.raises(RadarAssimilationError, match="z_source"):
        _config(z_source="z_mean")


def test_config_refuses_unknown_fall_speed():
    with pytest.raises(RadarAssimilationError, match="fall_speed"):
        _config(fall_speed="always")
    assert set(FALL_SPEED_POLICIES) == {"none", "reflectivity"}


def test_reflectivity_without_provider_is_a_refusal(world, grid):
    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in world.member_states.items()}
    with pytest.raises(RadarAssimilationError,
                       match="reflectivity_provider"):
        assimilate_radar_grid(checkpoints, world.obs_path, grid,
                              _config(reflectivity=True,
                                      analysis_fields=HYDRO_FIELDS,
                                      positivity_policy="clip"))
    with pytest.raises(RadarAssimilationError,
                       match="reflectivity_provider"):
        assimilate_radar_grid(checkpoints, world.obs_path, grid,
                              _config(fall_speed="reflectivity"))


def test_reflectivity_against_a_wind_only_state_vector_is_refused():
    """The ablation's own finding, turned into a refusal.

    Reflectivity constrains condensate.  With ``u``/``v`` alone in the
    state vector every dBZ increment would come from wind-hydrometeor
    sampling covariance in a rank-``R-1`` ensemble, which is noise, and
    the run would report a reflectivity analysis that analysed nothing.
    """
    with pytest.raises(RadarAssimilationError,
                       match="no analysed field is a thermodynamic"):
        _config(reflectivity=True, analysis_fields=("u", "v"))


def test_constrained_field_without_a_positivity_policy_is_refused():
    with pytest.raises(RadarAssimilationError,
                       match="states no positivity_policy"):
        _config(analysis_fields=("u", "v", "qr"))
    # ... and an unconstrained field set does not need one.
    assert _config(analysis_fields=("u", "v")).positivity_policy is None


def test_config_refuses_unknown_relaxation_and_moment_policy():
    with pytest.raises(RadarAssimilationError, match="relaxation must be"):
        _config(relaxation="rtpq")
    with pytest.raises(RadarAssimilationError, match="moment_policy must be"):
        _config(moment_policy="whatever")


def test_mapping_bound_observations_refuse_unknown_cycle(world, grid):
    assimilate = make_assimilate({0: world.obs_path}, grid, _config())
    with pytest.raises(RadarAssimilationError, match="cycle 3"):
        assimilate(3, world.member_states)


# ---------------------------------------------------------------------------
# the analysis, end to end on the driver's own seam shapes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analysis(world, grid):
    assimilate = make_assimilate(world.obs_path, grid, _config())
    increments, provenance = assimilate(0, world.member_states)
    return increments, provenance


def test_increments_are_seam_shaped(analysis, world):
    increments, _ = analysis
    assert set(increments) == set(world.member_states)
    for index, member in increments.items():
        assert set(member) == {"u", "v"}
        assert member["u"].shape == (NZ, NY, NX + 1)
        assert member["v"].shape == (NZ, NY + 1, NX)
        for name in ("u", "v"):
            assert np.all(np.isfinite(member[name]))


def test_increments_zero_outside_localization(analysis):
    """The far corner sits outside every observation's lens; face points
    there average mass increments that are exactly zero, so they are
    exactly zero too."""
    increments, _ = analysis
    corner = 2
    saw_signal = 0.0
    for member in increments.values():
        for name in ("u", "v"):
            block = member[name][:, :corner, :corner]
            assert np.all(block == 0.0), (
                f"{name} increments in the unobserved corner are not "
                f"exactly zero (max {np.abs(block).max():g})")
            saw_signal = max(saw_signal, float(np.abs(member[name]).max()))
    assert saw_signal > 0.0, "every increment is zero; the check is vacuous"


def test_analysis_beats_prior_against_truth(analysis, world):
    """Ensemble-mean mass-point wind RMSE must drop where observed."""
    increments, _ = analysis
    region = world.interior
    for name in ("u", "v"):
        prior_members, post_members = [], []
        for index, info in world.member_states.items():
            state = read_checkpoint_state(
                member_background_checkpoint(info["member_dir"]))
            faces = np.asarray(state[name], np.float64)
            mass = (obsop.destagger_u(faces) if name == "u"
                    else obsop.destagger_v(faces))
            inc_faces = np.asarray(increments[index][name], np.float64)
            inc_mass = (obsop.destagger_u(inc_faces) if name == "u"
                        else obsop.destagger_v(inc_faces))
            prior_members.append(mass)
            post_members.append(mass + inc_mass)
        prior_error = np.mean(prior_members, axis=0) - world.truth[name]
        post_error = np.mean(post_members, axis=0) - world.truth[name]
        prior_rmse = float(np.sqrt(np.mean(prior_error[region] ** 2)))
        post_rmse = float(np.sqrt(np.mean(post_error[region] ** 2)))
        assert post_rmse < prior_rmse, (
            f"{name}: analysis RMSE {post_rmse:.3f} did not improve on "
            f"prior {prior_rmse:.3f}")


def test_provenance_names_what_happened(analysis):
    _, provenance = analysis
    assert provenance["schema"] == "gpuwm-da.radar-assimilation.v1"
    assert provenance["stability"] == "experimental"
    assert provenance["cycle"] == 0
    assert provenance["members"] == MEMBERS
    assert provenance["wind_fields_restaggered"] == ["u", "v"]
    assert provenance["fall_speed"] == "none"
    names = {entry["name"] for entry in provenance["innovations"]}
    assert names == {"vr:AAAA", "vr:BBBB"}
    for entry in provenance["innovations"]:
        assert entry["observations"] > 0
        assert entry["innovation_rms"] > 0.0
        assert np.isfinite(entry["innovation_mean"])
        assert entry["obs_error_mean"] == pytest.approx(OBS_ERR_MS)
    assert provenance["filter"]["active_points"] > 0
    # JSON-serialisable, because it lands in the cycle manifest.
    import json
    json.dumps(provenance)


def test_increments_apply_to_the_real_checkpoint(analysis, world, tmp_path):
    from gpuwm.ensemble.increments import apply_increments_to_checkpoint

    increments, _ = analysis
    index = sorted(world.member_states)[0]
    background = member_background_checkpoint(
        world.member_states[index]["member_dir"])
    receipt = apply_increments_to_checkpoint(
        background, increments[index], tmp_path / "analysis.npz")
    assert receipt["field_count"] == 2
    assert receipt["state_sha256_before"] != receipt["state_sha256_after"]
    analysed = read_checkpoint_state(tmp_path / "analysis.npz")
    original = read_checkpoint_state(background)
    for name in ("u", "v"):
        expected = (original[name]
                    + increments[index][name].astype(np.float32))
        assert np.allclose(analysed[name], expected, atol=1e-6)


def test_checkpoints_from_a_different_domain_are_refused(world, grid,
                                                         tmp_path):
    member = tmp_path / "member_000"
    member.mkdir()
    _write_checkpoint(member / "gpuwmrst_d01_000600.npz", {
        "u": np.zeros((NZ, NY, NX + 3), np.float32),
        "v": np.zeros((NZ, NY + 3, NX + 2), np.float32),
        "w": np.zeros((NZ + 1, NY + 2, NX + 2), np.float32),
    })
    checkpoints = dict.fromkeys(
        range(2), member / "gpuwmrst_d01_000600.npz")
    with pytest.raises(RadarAssimilationError, match="not from this"):
        assimilate_radar_grid(checkpoints, world.obs_path, grid, _config())


# ---------------------------------------------------------------------------
# scheme reflectivity provider and the fall-speed closure
# ---------------------------------------------------------------------------


def test_scheme_reflectivity_provider_kessler(world):
    from gpuwm.config import RunConfig

    run = RunConfig(nx=NX, ny=NY, nz=NZ, dx=DX_M, dy=DX_M, dt=10.0,
                    run_seconds=600.0, ztop=TOP_M, mp_physics=1)
    provider = scheme_reflectivity_provider(
        run, base_theta=np.linspace(300.0, 330.0, NZ))
    info = world.member_states[0]
    state = read_checkpoint_state(
        member_background_checkpoint(info["member_dir"]))
    dbz = provider(0, state)
    assert dbz.shape == (NZ, NY, NX)
    assert np.all(np.isfinite(dbz))
    # Rain is present, so somewhere must read above the clear-air floor.
    assert float(dbz.max()) > -35.0


def test_scheme_provider_refuses_bad_base_theta():
    from gpuwm.config import RunConfig

    run = RunConfig(nx=NX, ny=NY, nz=NZ, dx=DX_M, dy=DX_M, dt=10.0,
                    run_seconds=600.0, ztop=TOP_M, mp_physics=1)
    with pytest.raises(RadarAssimilationError, match="base_theta"):
        scheme_reflectivity_provider(run, base_theta=np.zeros((2, 2)))


def test_full_operator_with_fall_speed(world, grid):
    """Vr with the Sun & Crook closure wired through the scheme's dBZ."""
    from gpuwm.config import RunConfig

    run = RunConfig(nx=NX, ny=NY, nz=NZ, dx=DX_M, dy=DX_M, dt=10.0,
                    run_seconds=600.0, ztop=TOP_M, mp_physics=1)
    provider = scheme_reflectivity_provider(
        run, base_theta=np.linspace(300.0, 330.0, NZ))
    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in world.member_states.items()}
    increments, provenance = assimilate_radar_grid(
        checkpoints, world.obs_path, grid,
        _config(fall_speed="reflectivity"),
        reflectivity_provider=provider)
    assert provenance["fall_speed"] == "reflectivity"
    for member in increments.values():
        for name in ("u", "v"):
            assert np.all(np.isfinite(member[name]))


# ---------------------------------------------------------------------------
# observation thinning and error inflation
# ---------------------------------------------------------------------------


def test_thin_mask_picks_most_gates_then_smallest_error():
    from gpuwm.da.radar_assimilation import thin_mask

    mask = np.ones((1, 4, 4), bool)
    counts = np.zeros((1, 4, 4))
    errors = np.ones((1, 4, 4))
    # Block (0,0): counts decide.
    counts[0, 1, 1] = 9.0
    # Block (0,1): counts tie at 0, errors decide.
    errors[0, 0, 3] = 0.25
    # Block (1,0): all equal -> first cell in block order wins.
    # Block (1,1): no observations at all.
    mask[0, 2:, 2:] = False
    kept = thin_mask(mask, counts, errors, 2)
    assert kept.sum() == 3
    assert kept[0, 1, 1]          # most gates
    assert kept[0, 0, 3]          # smallest error
    assert kept[0, 2, 0]          # first cell of an all-tied block
    assert not kept[0, 2:, 2:].any()   # empty block keeps none
    # Survivors are always original observations.
    assert not (kept & ~mask).any()


def test_thin_mask_identity_and_refusals():
    from gpuwm.da.radar_assimilation import thin_mask

    mask = np.zeros((2, 3, 5), bool)
    mask[0, 1, 2] = True
    kept = thin_mask(mask, np.ones(mask.shape), np.ones(mask.shape), 1)
    assert np.array_equal(kept, mask)
    # Non-divisible extents are handled by padding, not refused.
    kept3 = thin_mask(mask, np.ones(mask.shape), np.ones(mask.shape), 3)
    assert kept3.sum() == 1 and kept3[0, 1, 2]
    with pytest.raises(RadarAssimilationError):
        thin_mask(mask, np.ones(mask.shape), np.ones(mask.shape), 0)
    with pytest.raises(RadarAssimilationError):
        thin_mask(mask[0], np.ones((3, 5)), np.ones((3, 5)), 2)


def test_config_refuses_bad_thinning_inflation_device():
    with pytest.raises(RadarAssimilationError, match="thinning"):
        _config(velocity_thinning_cells=0)
    with pytest.raises(RadarAssimilationError, match="inflation"):
        _config(velocity_error_inflation=0.5)
    with pytest.raises(RadarAssimilationError, match="solve_device"):
        _config(solve_device="gpu")


def test_thinned_inflated_analysis_end_to_end(world, grid):
    """Thinning caps the local observation count; inflation reaches the
    batches the filter consumed; the structural zeros survive."""
    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in world.member_states.items()}
    base_cfg = _config()
    _, base_prov = assimilate_radar_grid(
        checkpoints, world.obs_path, grid, base_cfg)
    thinned_cfg = _config(velocity_thinning_cells=2,
                          velocity_error_inflation=2.0)
    increments, provenance = assimilate_radar_grid(
        checkpoints, world.obs_path, grid, thinned_cfg)

    receipt = provenance["velocity_thinning"]
    assert receipt["cells"] == 2
    assert receipt["error_inflation"] == 2.0
    for radar in receipt["radars"]:
        # One survivor per non-empty 2x2 block: at least a quarter of the
        # observations survive, and strictly fewer than all of them.
        assert (radar["points_before"] / 4.0
                <= radar["points_after"] < radar["points_before"])
    for entry in provenance["innovations"]:
        assert entry["obs_error_mean"] == pytest.approx(2.0 * OBS_ERR_MS)
    assert (provenance["filter"]["max_local_obs"]
            < base_prov["filter"]["max_local_obs"])
    corner = 2
    for member in increments.values():
        for name in ("u", "v"):
            assert np.all(np.isfinite(member[name]))
            assert np.all(member[name][:, :corner, :corner] == 0.0)


# ---------------------------------------------------------------------------
# innovation summary arithmetic
# ---------------------------------------------------------------------------


def test_innovation_summary_hand_computed():
    from gpuwm.da.letkf import GriddedObs

    values = np.zeros((1, 1, 2))
    values[0, 0, 0] = 2.0
    mask = np.zeros((1, 1, 2), bool)
    mask[0, 0, 0] = True
    simulated = np.zeros((2, 1, 1, 2))
    simulated[0, 0, 0, 0] = 1.0
    simulated[1, 0, 0, 0] = 3.0     # mean H(x) = 2.0 -> innovation 0
    errors = np.full((1, 1, 2), 0.5)
    batch = GriddedObs(name="vr:TEST", values=values, errors=errors,
                       simulated=simulated, mask=mask)
    (entry,) = innovation_summary([batch])
    assert entry["observations"] == 1
    assert entry["obs_mean"] == 2.0
    assert entry["hx_mean"] == 2.0
    assert entry["innovation_mean"] == 0.0
    assert entry["innovation_rms"] == 0.0
    assert entry["ensemble_spread_mean"] == pytest.approx(np.sqrt(2.0))
    assert entry["obs_error_mean"] == 0.5


def test_innovation_summary_empty_batch_reports_count_only():
    from gpuwm.da.letkf import GriddedObs

    shape = (1, 2, 2)
    batch = GriddedObs(name="vr:NONE", values=np.zeros(shape),
                       errors=np.ones(shape),
                       simulated=np.zeros((2,) + shape),
                       mask=np.zeros(shape, bool))
    (entry,) = innovation_summary([batch])
    assert entry == {"name": "vr:NONE", "observations": 0}


# ---------------------------------------------------------------------------
# the moisture / hydrometeor analysis
#
# The same twin-experiment gate the wind analysis is held to, extended to
# the variables reflectivity actually constrains.  The ensemble is built
# by the REAL producer -- gpuwm.da.perturb, not a hand-rolled draw in this
# file -- because the whole point of the milestone is that a perturbation
# module and a filter have to agree about what a two-moment state is, and
# a fixture that perturbs its own way could agree with neither.
# ---------------------------------------------------------------------------

HZ, HY, HX = 6, 24, 24
H_DX_M = 3000.0
H_MEMBERS = 12
#: Resolvable on this domain: >= 2 grid spacings and <= span/(2*pi).
H_SCALE_KM = 8.0
H_RIM = 3
H_MARGIN = 8
Z_ERR_DBZ = 5.0


def _morrison_run_config():
    from gpuwm.config import RunConfig

    return RunConfig(nx=HX, ny=HY, nz=HZ, dx=H_DX_M, dy=H_DX_M, dt=10.0,
                     run_seconds=600.0, ztop=TOP_M, mp_physics=10)


def _hydro_center():
    """A background with one storm: condensate in a blob, clear outside.

    Deliberately zero outside the blob.  A multiplicative perturbation
    leaves clear air exactly clear, so this is also the fixture that
    proves the filter cannot create echo where no member has any -- the
    honest limitation the milestone has to state.
    """
    base_theta = np.linspace(300.0, 340.0, HZ)
    column = np.linspace(9.2e4, 3.0e4, HZ)
    zz, yy, xx = np.meshgrid(np.arange(HZ), np.arange(HY), np.arange(HX),
                             indexing="ij")
    radius = np.sqrt((yy - 12.0) ** 2 + (xx - 12.0) ** 2)
    blob = np.exp(-(radius / 4.0) ** 2) * np.exp(-((zz - 2.5) / 2.0) ** 2)
    state = {
        "u": np.full((HZ, HY, HX + 1), 6.0),
        "v": np.full((HZ, HY + 1, HX), -3.0),
        "w": np.zeros((HZ + 1, HY, HX)),
        "thp": np.zeros((HZ, HY, HX)),
        "qv": np.full((HZ, HY, HX), 6.0e-3),
        "p": np.broadcast_to(column[:, None, None], (HZ, HY, HX)).copy(),
        "qr": 1.2e-3 * blob,
        "nr": 8.0e3 * blob,
        "qs": 4.0e-4 * blob,
        "ns": 4.0e3 * blob,
        "qg": 6.0e-4 * blob,
        "ng": 2.0e3 * blob,
    }
    # Below the scheme's own activity gate the species is absent, and a
    # background that trails off to 1e-30 is not a storm edge, it is
    # noise the guard would have to reason about.
    for name, number in (("qr", "nr"), ("qs", "ns"), ("qg", "ng")):
        off = state[name] < 1.0e-8
        state[name] = np.where(off, 0.0, state[name])
        state[number] = np.where(off, 0.0, state[number])
    return state, base_theta


def _perturbed_member(center, base_theta, seed):
    """One member, perturbed by gpuwm.da.perturb's own contract."""
    from gpuwm.da import perturb

    state = types.SimpleNamespace(**{name: values.copy()
                                     for name, values in center.items()})
    state.thb = base_theta
    for absent in ("qc", "nc", "qi", "ni", "qh", "php", "mup", "al", "alt",
                   "h_diabatic", "qndrop", "qvolg", "qvolh"):
        setattr(state, absent, None)
    cfg = perturb.PerturbationConfig(
        dx_km=H_DX_M / 1000.0, dy_km=H_DX_M / 1000.0, rim_width=H_RIM,
        fields=(
            perturb.FieldPerturbation("u", 1.5, H_SCALE_KM),
            perturb.FieldPerturbation("v", 1.5, H_SCALE_KM),
            perturb.FieldPerturbation("theta", 0.5, H_SCALE_KM),
            perturb.FieldPerturbation("qv", 0.05, H_SCALE_KM,
                                      mode="lognormal", clip_sigmas=2.5),
        ),
        species=(
            perturb.SpeciesPerturbation("qr", 0.7, H_SCALE_KM,
                                        clip_sigmas=2.5),
            perturb.SpeciesPerturbation("qs", 0.7, H_SCALE_KM,
                                        clip_sigmas=2.5),
            perturb.SpeciesPerturbation("qg", 0.7, H_SCALE_KM,
                                        clip_sigmas=2.5),
        ))
    provenance = perturb.apply_perturbations(state, seed, cfg)
    fields = {name: getattr(state, name)
              for name in ("u", "v", "w", "thp", "qv", "p",
                           "qr", "nr", "qs", "ns", "qg", "ng")}
    return fields, provenance


@pytest.fixture(scope="module")
def hydro_grid():
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid

    projection = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.0, dx=H_DX_M, dy=H_DX_M, e_we=HX + 1, e_sn=HY + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, TOP_M, HZ + 1),
        name="radar-assim-hydro")


@pytest.fixture(scope="module")
def hydro_world(hydro_grid, tmp_path_factory):
    """Twin experiment: truth is one more draw from the same centre."""
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.superob import GriddedObservations, SuperobParams

    root = tmp_path_factory.mktemp("radar-assim-hydro")
    center, base_theta = _hydro_center()
    run = _morrison_run_config()
    provider = scheme_reflectivity_provider(run, base_theta=base_theta)

    truth, _ = _perturbed_member(center, base_theta, 9000001)

    leg_root = root / "cycle_000"
    member_states = {}
    provenances = []
    for index in range(H_MEMBERS):
        member_dir = leg_root / f"member_{index:03d}"
        member_dir.mkdir(parents=True)
        fields, provenance = _perturbed_member(center, base_theta,
                                               7000000 + index)
        _write_checkpoint(member_dir / "gpuwmrst_d01_000600.npz", fields)
        member_states[index] = {"member_dir": str(member_dir)}
        provenances.append(provenance)

    shape = (HZ, HY, HX)
    interior = np.zeros(shape, bool)
    interior[:, H_MARGIN:HY - H_MARGIN, H_MARGIN:HX - H_MARGIN] = True

    # -- reflectivity observations, from the truth through the SAME H(x)
    rng = np.random.default_rng(4242)
    z_truth = provider(0, truth)
    z_mask = interior & (z_truth > -10.0)
    z_obs = np.where(z_mask, z_truth + rng.normal(0.0, Z_ERR_DBZ, shape),
                     0.0)

    # -- radial velocity, exactly as the wind fixture does it ------------
    sites = [obsop.RadarSite(latitude_deg=float(hydro_grid.lat[HY // 2, 1]),
                             longitude_deg=float(hydro_grid.lon[HY // 2, 1]),
                             altitude_m=350.0, name="AAAA")]
    geometry = obsop.GridGeometry.from_target_grid(hydro_grid)
    sina, cosa = grid_rotation(hydro_grid)
    u_e, v_n = obsop.earth_relative_winds(
        obsop.destagger_u(truth["u"]), obsop.destagger_v(truth["v"]),
        sina, cosa)
    w_mass = obsop.destagger_w(truth["w"])
    beam = obsop.beam_geometry(geometry, sites[0])
    east, north, up = (np.broadcast_to(np.asarray(c, np.float64),
                                       shape).copy()
                       for c in beam.unit_vector_enu())
    vr_obs = (u_e * east + v_n * north + w_mass * up
              + rng.normal(0.0, OBS_ERR_MS, shape))[None]
    vr_mask = interior[None].astype(np.int8)

    observations = GriddedObservations(
        z_obs=z_obs.astype(np.float32), z_mask=z_mask.astype(np.int8),
        z_err=np.where(z_mask, Z_ERR_DBZ, 0.0).astype(np.float32),
        z_max=z_obs.astype(np.float32), z_mean=z_obs.astype(np.float32),
        z_count=z_mask.astype(np.int32),
        vr_obs=vr_obs.astype(np.float32), vr_mask=vr_mask,
        vr_err=np.where(vr_mask.astype(bool), OBS_ERR_MS,
                        0.0).astype(np.float32),
        vr_count=vr_mask.astype(np.int32),
        vr_rejected=np.zeros((1,) + shape, np.int32),
        vr_beam_east=east[None].astype(np.float32),
        vr_beam_north=north[None].astype(np.float32),
        vr_beam_up=up[None].astype(np.float32),
        radars=[{"id": sites[0].name, "lat_deg": sites[0].latitude_deg,
                 "lon_deg": sites[0].longitude_deg,
                 "alt_m": sites[0].altitude_m,
                 "valid_time": "2026-01-01T00:00:00Z"}],
        counts=[], provenance=[])
    obs_path = root / "obs-radar-grid.nc"
    write_radar_grid(obs_path, observations, hydro_grid,
                     valid_time="2026-01-01T00:00:00Z",
                     params=SuperobParams(), overwrite=True)

    return types.SimpleNamespace(
        root=root, center=center, base_theta=base_theta, truth=truth,
        member_states=member_states, obs_path=obs_path, interior=interior,
        z_mask=z_mask, provider=provider, run=run,
        perturbation=provenances)


def test_perturbation_gives_every_species_spread_and_breaks_no_pair(
        hydro_world):
    """Criterion (a), at the source: the ensemble carries the spread."""
    from gpuwm.da.moments import moment_consistency_report

    members = [read_checkpoint_state(
        member_background_checkpoint(info["member_dir"]))
        for info in hydro_world.member_states.values()]
    for name in ("qv", "qr", "qs", "qg", "nr", "ns", "ng", "thp"):
        stack = np.stack([m[name] for m in members]).astype(np.float64)
        spread = stack.std(axis=0, ddof=1)
        assert float(spread.max()) > 0.0, f"{name} has no ensemble spread"
    # ... and every member is still a state the scheme can evaluate.
    for member in members:
        report = moment_consistency_report(member, mp_physics=10)
        assert report["consistent"], report
        assert report["offending_cells_total"] == 0
        assert report["nonfinite_cells_total"] == 0
    # The species records say so themselves, per member.
    for provenance in hydro_world.perturbation:
        assert provenance["species"], "no species were perturbed"
        for record in provenance["species"]:
            assert record["depleted_pairs_created"] == 0
            assert record["negative_points"] == 0
            assert record["active_points"] > 0
            assert record["factor_min"] > 0.0


def _hydro_config(**overrides):
    from gpuwm.da.moments import analysis_fields

    carried = ("u", "v", "w", "thp", "qv", "p",
               "qr", "nr", "qs", "ns", "qg", "ng")
    fields = tuple(name for name in analysis_fields(10) if name in carried)
    kwargs = dict(
        localization=Localization(horizontal_m=H_LOC_M, vertical_m=V_LOC_M),
        rtps_alpha=0.9, analysis_fields=fields, velocity=True,
        reflectivity=True, positivity_policy="clip", mp_physics=10)
    kwargs.update(overrides)
    return RadarAssimilationConfig(**kwargs)


@pytest.fixture(scope="module")
def hydro_analysis(hydro_world, hydro_grid):
    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in hydro_world.member_states.items()}
    return assimilate_radar_grid(
        checkpoints, hydro_world.obs_path, hydro_grid, _hydro_config(),
        reflectivity_provider=hydro_world.provider)


def test_hydro_analysis_updates_the_scheme_whole_moment_set(hydro_analysis):
    increments, provenance = hydro_analysis
    analysed = set(provenance["analysis_fields"])
    assert {"thp", "qv", "u", "v", "qr", "nr", "qs", "ns", "qg",
            "ng"} <= analysed
    receipt = provenance["moment_policy"]
    assert receipt["policy"] == "full-moment"
    assert receipt["repair_required"] is False
    assert set(receipt["pairs_updated"]) == {"qr", "qs", "qg"}
    assert receipt["mass_only_species"] == []


def test_hydro_analysis_beats_the_prior_against_truth(hydro_world,
                                                      hydro_analysis):
    """The twin-experiment skill gate, on the new variables.

    Scored inside the observed region only: outside it the filter is
    contractually the identity, so including it would dilute the very
    quantity being measured with points nobody claimed to improve.
    """
    increments, _ = hydro_analysis
    members = {index: read_checkpoint_state(
        member_background_checkpoint(info["member_dir"]))
        for index, info in hydro_world.member_states.items()}
    inside = hydro_world.interior
    improved = {}
    for name in ("qr", "qs", "qg", "qv"):
        truth = hydro_world.truth[name]
        prior = np.stack([members[i][name] for i in sorted(members)])
        analysis = prior + np.stack([increments[i][name]
                                     for i in sorted(increments)])
        prior_err = float(np.sqrt(np.mean(
            (prior.mean(axis=0) - truth)[inside] ** 2)))
        post_err = float(np.sqrt(np.mean(
            (analysis.mean(axis=0) - truth)[inside] ** 2)))
        improved[name] = (prior_err, post_err)
    # Rain is what the reflectivity operator is most sensitive to and is
    # the species the gate is asserted on; the rest are reported so a
    # regression shows which one moved.
    prior_err, post_err = improved["qr"]
    assert post_err < prior_err, improved


def test_hydro_increments_are_exactly_zero_outside_localization(
        hydro_analysis):
    """The guarantee that has survived every run so far, on 10 fields.

    Not "small" -- the literal ``0.0``.  With prior_inflation = 1 the
    inactive-point transform is the exact identity, so a corner outside
    every observation's lens must come back bitwise unchanged for every
    analysed field, hydrometeors included.
    """
    increments, provenance = hydro_analysis
    for member in increments.values():
        for name, values in member.items():
            corner = values[:, :2, :2]
            assert np.array_equal(corner, np.zeros_like(corner)), name
    # And every mass-shaped field reached exactly the same gridpoints.
    mass = [n for n in provenance["analysis_fields"] if n not in ("u", "v")]
    support = None
    for name in mass:
        stack = np.stack([increments[i][name] for i in sorted(increments)])
        touched = np.any(stack != 0.0, axis=0)
        if support is None:
            support = touched
        else:
            assert np.array_equal(support, touched), name
    assert int(support.sum()) < support.size


def test_hydro_analysis_is_non_negative_and_moment_consistent(
        hydro_world, hydro_analysis):
    """Criterion (d): the analysed state is one the scheme can evaluate."""
    from gpuwm.da.moments import moment_consistency_report

    increments, provenance = hydro_analysis
    positivity = provenance["positivity"]
    assert positivity["policy"] == "clip"
    assert positivity["mass_left_negative"] == 0.0
    for index, member in increments.items():
        background = read_checkpoint_state(member_background_checkpoint(
            hydro_world.member_states[index]["member_dir"]))
        analysis = dict(background)
        for name, values in member.items():
            analysis[name] = np.asarray(background[name],
                                        np.float64) + values
        for name in ("qv", "qr", "qs", "qg", "nr", "ns", "ng"):
            assert float(np.min(analysis[name])) >= 0.0, (index, name)
            assert np.all(np.isfinite(analysis[name])), (index, name)
        report = moment_consistency_report(analysis, mp_physics=10)
        assert report["consistent"], (index, report)


def test_hydro_analysis_applies_through_the_shipped_applier(
        hydro_world, hydro_analysis, tmp_path):
    """The applier's own moment guard has nothing to repair."""
    from gpuwm.ensemble.increments import apply_increments_to_checkpoint

    increments, _ = hydro_analysis
    source = member_background_checkpoint(
        hydro_world.member_states[0]["member_dir"])
    receipt = apply_increments_to_checkpoint(
        source, increments[0], tmp_path / "analysis.npz", mp_physics=10)
    assert receipt["moments"]["consistent"] is True
    assert receipt["moments"]["offending_cells_total"] == 0
    assert receipt["moments"]["nonfinite_cells_total"] == 0
    assert receipt["moments"]["repaired_cells_total"] == 0


def test_hydro_prior_and_posterior_spread_are_both_reported(hydro_analysis):
    """The instrument the spread diagnosis needs, present and paired."""
    _, provenance = hydro_analysis
    filt = provenance["filter"]
    assert set(filt["prior_spread"]) == set(filt["posterior_spread"])
    assert set(filt["prior_spread"]) == set(provenance["analysis_fields"])
    for name, prior in filt["prior_spread"].items():
        assert prior > 0.0, name
    assert filt["relaxation"] == "rtps"
    assert filt["rtps_alpha"] == 0.9


def test_the_receipt_names_the_eigensolver_that_produced_it(analysis):
    """A cycle report has to say which solver factored its matrices.

    The bundled Jacobi kernel is the default, so a device run stopped
    calling cuSOLVER the moment that landed.  The two agree to ~1e-11
    relative and NOT bitwise, which means every receipt banked before the
    change reproduces only under ``eigensolver='library'``.  A receipt that
    does not name its solver cannot be reproduced on purpose.

    This fixture solves on the host, where the kernel does not apply, so
    the honest answer here is the library solver and no sweeps.
    """
    _, provenance = analysis
    filt = provenance["filter"]
    assert filt["eigensolver"] == "library"
    assert filt["max_jacobi_sweeps"] == 0


def test_rtpp_is_available_and_differs_from_rtps(hydro_world, hydro_grid):
    """Both relaxations run; at the same alpha they are not the same.

    The ensemble MEAN increment is relaxation-independent -- neither
    scheme touches ``wbar`` -- while the individual members are not.
    That pair of facts is what makes the knob a spread knob rather than
    a second analysis.
    """
    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in hydro_world.member_states.items()}
    rtpp, prov = assimilate_radar_grid(
        checkpoints, hydro_world.obs_path, hydro_grid,
        _hydro_config(relaxation="rtpp"),
        reflectivity_provider=hydro_world.provider)
    assert prov["relaxation"] == "rtpp"
    assert prov["filter"]["relaxation"] == "rtpp"
    rtps, _ = assimilate_radar_grid(
        checkpoints, hydro_world.obs_path, hydro_grid, _hydro_config(),
        reflectivity_provider=hydro_world.provider)
    mean_rtpp = np.mean([rtpp[i]["thp"] for i in sorted(rtpp)], axis=0)
    mean_rtps = np.mean([rtps[i]["thp"] for i in sorted(rtps)], axis=0)
    assert np.allclose(mean_rtpp, mean_rtps, atol=1e-10)
    assert not np.allclose(rtpp[0]["thp"], rtps[0]["thp"], atol=1e-10)


def test_reflectivity_thinning_reduces_the_assimilated_count(hydro_world,
                                                             hydro_grid):
    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in hydro_world.member_states.items()}
    _, provenance = assimilate_radar_grid(
        checkpoints, hydro_world.obs_path, hydro_grid,
        _hydro_config(reflectivity_thinning_cells=2,
                      reflectivity_error_inflation=3.0),
        reflectivity_provider=hydro_world.provider)
    receipt = provenance["reflectivity_thinning"]
    assert receipt["cells"] == 2
    assert receipt["error_inflation"] == 3.0
    assert 0 < receipt["points_after"] < receipt["points_before"]
    (z_batch,) = [entry for entry in provenance["innovations"]
                  if entry["name"] == "z"]
    assert z_batch["observations"] == receipt["points_after"]
    assert z_batch["obs_error_mean"] == pytest.approx(Z_ERR_DBZ * 3.0)


# ---------------------------------------------------------------------------
# the guards, watched FIRING
#
# The tests above watch the moment and positivity guards stay quiet on a
# healthy analysis, which is the outcome the design is for and is also
# the outcome a guard that had been accidentally disabled would produce.
# These four make each one fire on purpose and check the number it
# reports, so "clean" upstream means "clean" and not "asleep".
# ---------------------------------------------------------------------------


def test_positivity_clip_fires_and_is_counted(hydro_world, hydro_grid):
    """Clip at zero ADDS mass; the count and the mass are the diagnostic.

    Forced by shrinking the observation error until the filter proposes
    increments larger than the background it is correcting, which is
    exactly the regime a badly calibrated sigma_o puts a real run in.
    The point is not that the analysis is good -- it is not -- but that
    the policy fires, is counted, and leaves nothing negative behind.
    """
    from gpuwm.da.positivity import apply_positivity, verify_non_negative

    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in hydro_world.member_states.items()}
    increments, provenance = assimilate_radar_grid(
        checkpoints, hydro_world.obs_path, hydro_grid,
        _hydro_config(reflectivity_error_inflation=1.0,
                      positivity_policy="none"),
        reflectivity_provider=hydro_world.provider)
    # With no policy, negatives are counted and LEFT -- a stated choice.
    receipt = provenance["positivity"]
    assert receipt is None or receipt["policy"] == "none"

    prior = {}
    for name in ("qr", "qs", "qg", "qv"):
        prior[name] = np.stack([
            read_checkpoint_state(member_background_checkpoint(
                hydro_world.member_states[i]["member_dir"]))[name]
            for i in sorted(hydro_world.member_states)]).astype(np.float64)
    stacked = {name: np.stack([increments[i][name]
                               for i in sorted(increments)])
               for name in prior}
    negatives = sum(int(np.count_nonzero(prior[n] + stacked[n] < 0.0))
                    for n in prior)
    assert negatives > 0, (
        "this fixture no longer drives any analysis negative, so it can no "
        "longer prove the clip fires; tighten sigma_o further")

    clipped, clip_receipt = apply_positivity(prior, stacked, policy="clip")
    assert clip_receipt["negative_points"] == negatives
    assert clip_receipt["mass_added_by_clip"] > 0.0
    assert clip_receipt["mass_left_negative"] == 0.0
    verify_non_negative(prior, clipped)
    for name in prior:
        assert float((prior[name] + clipped[name]).min()) >= 0.0
    # ... and the wetward bias is real: clipping only ever adds.
    for entry in clip_receipt["per_field"]:
        assert entry["mass_added_by_clip"] >= 0.0


def test_moment_guard_refuses_a_truncated_field_set_before_the_solve(
        hydro_world, hydro_grid):
    """A mass field without its number moment, refused at configuration.

    This is the node-8 defect: nine fields, no number concentrations,
    reflectivity assimilation creating rain in cells the background left
    clear, and Morrison's slope closure evaluating to NaN.  Under
    full-moment it is now refused before a single H(x) is computed,
    naming every species.
    """
    from gpuwm.da.moments import MomentPolicyError

    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in hydro_world.member_states.items()}
    truncated = ("u", "v", "thp", "qv", "qr", "qs", "qg")   # no nr/ns/ng
    with pytest.raises(MomentPolicyError, match="qr without nr"):
        assimilate_radar_grid(
            checkpoints, hydro_world.obs_path, hydro_grid,
            _hydro_config(analysis_fields=truncated),
            reflectivity_provider=hydro_world.provider)


def test_a_multiplicative_ensemble_cannot_break_a_pair_even_mass_only(
        hydro_world, hydro_grid, tmp_path):
    """The truncation is legal once declared -- and here it is harmless.

    Under ``single-moment-with-repair`` the same field set the previous
    test refuses is allowed, and on a hand-built ensemble it is exactly
    how ``q > 0`` lands beside ``N = 0``.  On THIS ensemble it cannot,
    and the reason is a property of the perturbation rather than luck:
    the filter's increment lives in the span of the prior perturbations,
    a multiplicative perturbation is exactly zero where the species is
    absent, so every member agrees the cell is clear and the analysis
    increment there is bitwise zero.  A mass-only update can therefore
    only rescale condensate that already has a number moment.

    That is worth pinning, because it is the reason the run's
    ``moment_policy.repair_required`` is expected to stay False and not
    a claim that the guard is unnecessary -- the guard fires on a
    hand-built offender in ``tests/test_da_moments.py``, which is where
    the scheme's limiter is exercised directly.
    """
    from gpuwm.da.moments import moment_consistency_report
    from gpuwm.ensemble.increments import apply_increments_to_checkpoint

    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in hydro_world.member_states.items()}
    truncated = ("u", "v", "thp", "qv", "qr", "qs", "qg")
    increments, provenance = assimilate_radar_grid(
        checkpoints, hydro_world.obs_path, hydro_grid,
        _hydro_config(analysis_fields=truncated,
                      moment_policy="single-moment-with-repair",
                      reflectivity_error_inflation=1.0),
        reflectivity_provider=hydro_world.provider)
    receipt = provenance["moment_policy"]
    assert receipt["policy"] == "single-moment-with-repair"
    assert receipt["repair_required"] is True
    assert set(receipt["mass_only_species"]) == {"qr", "qs", "qg"}

    for index in sorted(increments):
        background = read_checkpoint_state(member_background_checkpoint(
            hydro_world.member_states[index]["member_dir"]))
        for mass, number in (("qr", "nr"), ("qs", "ns"), ("qg", "ng")):
            # Where every member is clear the increment is the literal
            # 0.0, so the mass-only update cannot promote a clear cell.
            clear = background[number] <= 0.0
            assert np.all(increments[index][mass][clear] == 0.0), (
                index, mass)
        applied = apply_increments_to_checkpoint(
            source_of := member_background_checkpoint(
                hydro_world.member_states[index]["member_dir"]),
            increments[index], tmp_path / f"a{index}.npz",
            moment_policy="single-moment-with-repair", mp_physics=10)
        assert applied["moments"]["offending_cells_total"] == 0, index
        assert applied["moments"]["nonfinite_cells_total"] == 0, index
        assert applied["moments"]["repaired_cells_total"] == 0, index
        assert str(source_of).endswith(".npz")

    # And the same field set with the repair switched off is therefore
    # NOT a refusal here: there is nothing broken to refuse.
    applied = apply_increments_to_checkpoint(
        member_background_checkpoint(
            hydro_world.member_states[0]["member_dir"]),
        increments[0], tmp_path / "no-repair.npz",
        moment_policy="single-moment-with-repair", moment_repair=False,
        mp_physics=10)
    assert applied["moments"]["consistent"] is True
    state = read_checkpoint_state(tmp_path / "no-repair.npz")
    assert moment_consistency_report(state, mp_physics=10)["consistent"]
