"""CPU suite for the Level-2 regional spectral operators.

The delivery record for this package (arwen-level2-regional-spectral-numerics,
2026-08-16) shipped an EMPTY combined patch: the operator sources arrived in
the repo overlay but the suites the construction container claimed to run
could not even be collected there (TEST-RESULTS.md shows the collection
error).  This file is therefore the first suite that actually runs the
operators, written against the delivered non-negotiable contracts:

- absent/off is bitwise inert and reads no state;
- shadow computes receipts but is state-bitwise inert;
- apply mutates only after every field and budget passed;
- periodic wrapping requires an explicit periodic-domain declaration;
- C-grid nonperiodic outer faces receive zero increment;
- the committed operator pin hash must not move.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from gpuwm.spectral_ops import (
    BandObservation, Hyperdiffusion, PINS, PINS_SHA256, RaisedCosineLowPass,
    ScalarTarget, SpectralBudgetExceeded, SpectralNumericsConfig, WindControl,
    apply_spectral_step, apply_transfer, blend_parent_child,
    damp_c_grid_divergence, damp_divergence, fit_hyperdiffusion, from_mapping,
    helmholtz_decompose, hyperdiffuse, nudge_large_scales, registration,
    split_scales, solve_helmholtz, solve_poisson,
)
from gpuwm.spectral_ops.pins import canonical_hash
from gpuwm.spectral_ops.receipt import (RECEIPT_SCHEMA, make_receipt,
                                        validate_receipt)
from gpuwm.spectral_ops.vector import (destagger_c_grid,
                                       lift_mass_increment_to_c_grid,
                                       spectral_divergence)

#: The delivered package's arithmetic identity, computed once from the
#: vendored sources at integration (2026-08-17) and committed.  The Level-2
#: contract says this hash MUST NOT MOVE: every step receipt embeds it, so a
#: drifted hash orphans every receipt already written.
COMMITTED_OPERATOR_PINS_SHA256 = (
    "549502b5f1b66fff4dda949ba5a16cfb9ed71bb52877c2ef2f395d36c031c2ad")

DX = 3000.0
DY = 3000.0


def wave(ny, nx, *, ky=0, kx=0, amplitude=1.0, dtype=np.float64):
    """One exact periodic Fourier mode on the mass grid."""
    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]
    return (amplitude * np.cos(2.0 * math.pi * (ky * y / ny + kx * x / nx))
            ).astype(dtype)


class ReadsNothing:
    """A model state that fails the test if anything reads it."""

    def __getattr__(self, name):  # pragma: no cover - the assertion itself
        raise AssertionError(f"inert spectral mode read state field {name!r}")


# ---------------------------------------------------------------------------
# pins


def test_committed_operator_pin_hash_has_not_moved():
    assert PINS_SHA256 == COMMITTED_OPERATOR_PINS_SHA256
    assert registration()["pins_sha256"] == COMMITTED_OPERATOR_PINS_SHA256


def test_pins_hash_is_canonical_and_tamper_evident():
    assert canonical_hash(PINS) == PINS_SHA256
    tampered = json.loads(json.dumps(PINS))
    tampered["default_mode"] = "apply"
    assert canonical_hash(tampered) != PINS_SHA256


def test_pins_default_mode_is_off():
    assert PINS["default_mode"] == "off"
    assert SpectralNumericsConfig().mode == "off"


# ---------------------------------------------------------------------------
# transfer functions


def test_hyperdiffusion_reference_wavelength_efolds_exactly():
    spec = Hyperdiffusion(order=3, reference_wavelength_m=6.0 * DX,
                          e_fold_time_s=300.0)
    k_ref = 2.0 * math.pi / spec.reference_wavelength_m
    value = spec.transfer(np.asarray([k_ref]), dt_s=300.0)
    assert value[0] == pytest.approx(math.exp(-1.0), rel=1e-12)


def test_hyperdiffusion_protected_scales_are_untouched():
    spec = Hyperdiffusion(order=3, reference_wavelength_m=6.0 * DX,
                          e_fold_time_s=300.0,
                          protect_wavelength_m=24.0 * DX)
    k_protected = 2.0 * math.pi / (48.0 * DX)
    k_reference = 2.0 * math.pi / spec.reference_wavelength_m
    values = spec.transfer(np.asarray([0.0, k_protected, k_reference]),
                           dt_s=600.0)
    assert values[0] == 1.0          # constant mode is a conservation law
    assert values[1] == 1.0          # at/below k_protect: zero activation
    assert values[2] < 1.0


def test_hyperdiffusion_damping_fraction_cap_is_a_floor_on_the_transfer():
    spec = Hyperdiffusion(order=2, reference_wavelength_m=4.0 * DX,
                          e_fold_time_s=1.0, maximum_damping_fraction=0.25)
    k = 2.0 * math.pi / (2.0 * DX)
    value = spec.transfer(np.asarray([k]), dt_s=1.0e6)
    assert value[0] == pytest.approx(0.75, abs=1e-12)


def test_hyperdiffusion_validation_refuses_nonsense():
    with pytest.raises(ValueError):
        Hyperdiffusion(order=0).validate()
    with pytest.raises(ValueError):
        Hyperdiffusion(e_fold_time_s=-1.0).validate()
    with pytest.raises(ValueError):
        Hyperdiffusion(reference_wavelength_m=6000.0,
                       protect_wavelength_m=6000.0).validate()
    with pytest.raises(ValueError):
        Hyperdiffusion(maximum_damping_fraction=1.5).validate()


# ---------------------------------------------------------------------------
# scalar operators


def test_periodic_hyperdiffusion_matches_the_exact_transfer():
    ny, nx = 48, 64
    spec = Hyperdiffusion(order=3, reference_wavelength_m=8.0 * DX,
                          e_fold_time_s=600.0)
    field = wave(ny, nx, kx=8)
    result = hyperdiffuse(field, dy_m=DY, dx_m=DX, dt_s=600.0, spec=spec,
                          boundary="periodic", periodic_domain=True,
                          preserve_mean=False)
    k = 2.0 * math.pi * 8 / (nx * DX)
    expected = field * spec.transfer(np.asarray([k]), dt_s=600.0)[0]
    np.testing.assert_allclose(result.values, expected, atol=1e-12)


def test_periodic_wrapping_requires_the_explicit_declaration():
    spec = Hyperdiffusion()
    with pytest.raises(ValueError, match="periodic_domain"):
        hyperdiffuse(wave(16, 16, kx=2), dy_m=DY, dx_m=DX, dt_s=60.0,
                     spec=spec, boundary="periodic")


def test_tapered_boundary_leaves_every_outer_edge_cell_unchanged():
    rng = np.random.default_rng(7)
    field = rng.normal(size=(40, 56))
    spec = Hyperdiffusion(order=3, reference_wavelength_m=6.0 * DX,
                          e_fold_time_s=120.0)
    result = hyperdiffuse(field, dy_m=DY, dx_m=DX, dt_s=120.0, spec=spec,
                          boundary="tapered", edge_taper_cells=8)
    np.testing.assert_array_equal(result.values[0, :], field[0, :])
    np.testing.assert_array_equal(result.values[-1, :], field[-1, :])
    np.testing.assert_array_equal(result.values[:, 0], field[:, 0])
    np.testing.assert_array_equal(result.values[:, -1], field[:, -1])
    assert result.rms_increment > 0.0


def test_tapered_boundary_preserves_the_mean_exactly_enough():
    rng = np.random.default_rng(11)
    field = rng.normal(size=(40, 56)) + 5.0
    spec = Hyperdiffusion(order=2, reference_wavelength_m=6.0 * DX,
                          e_fold_time_s=60.0)
    result = hyperdiffuse(field, dy_m=DY, dx_m=DX, dt_s=600.0, spec=spec,
                          boundary="tapered", edge_taper_cells=8)
    assert result.mean_after == pytest.approx(result.mean_before, rel=1e-12)


def test_reflect_boundary_damps_without_wrapping_artifacts():
    ny, nx = 32, 48
    y = np.arange(ny)[:, None] * np.ones((1, nx))
    ramp = y / ny  # discontinuous under periodic wrap, smooth under reflect
    spec = Hyperdiffusion(order=3, reference_wavelength_m=6.0 * DX,
                          e_fold_time_s=60.0)
    result = hyperdiffuse(ramp, dy_m=DY, dx_m=DX, dt_s=60.0, spec=spec,
                          boundary="reflect")
    # A reflect transform of an even-extended ramp must not ring at the
    # boundary the way a periodic wrap of the raw ramp would.
    assert result.max_abs_increment < 0.05
    assert result.mean_after == pytest.approx(result.mean_before, rel=1e-12)


def test_log_space_refuses_negative_input_and_respects_the_floor():
    spec = Hyperdiffusion(order=2, reference_wavelength_m=6.0 * DX,
                          e_fold_time_s=60.0)
    field = np.full((24, 24), 1.0e-6)
    field[8:16, 8:16] = 1.0e-3
    result = hyperdiffuse(field, dy_m=DY, dx_m=DX, dt_s=600.0, spec=spec,
                          boundary="reflect", space="log", floor=1.0e-9)
    assert result.minimum_after >= 1.0e-9
    with pytest.raises(ValueError, match="negative"):
        hyperdiffuse(field - 1.0, dy_m=DY, dx_m=DX, dt_s=600.0, spec=spec,
                     boundary="reflect", space="log", floor=1.0e-9)


def test_non_finite_operands_are_refused():
    field = wave(16, 16, kx=2)
    field[3, 3] = np.nan
    spec = Hyperdiffusion()
    with pytest.raises(ValueError, match="non-finite"):
        hyperdiffuse(field, dy_m=DY, dx_m=DX, dt_s=60.0, spec=spec,
                     boundary="reflect")


def test_apply_transfer_refuses_an_unknown_boundary_mode():
    with pytest.raises(ValueError, match="boundary"):
        apply_transfer(wave(16, 16, kx=1), dy_m=DY, dx_m=DX,
                       transfer_factory=lambda magnitude: magnitude * 0 + 1,
                       boundary="mirror")


# ---------------------------------------------------------------------------
# vector operators


def _divergent_flow(ny, nx):
    """phi = cos wave; u = dphi/dx, v = dphi/dy is purely divergent."""
    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]
    u = np.sin(2.0 * math.pi * 4 * x / nx) * np.ones((ny, 1))
    v = np.sin(2.0 * math.pi * 3 * y / ny) * np.ones((1, nx))
    return u, v


def test_divergence_damping_reduces_divergence_and_not_rotation():
    ny, nx = 48, 48
    u_div, v_div = _divergent_flow(ny, nx)
    spec = Hyperdiffusion(order=1, reference_wavelength_m=8.0 * DX,
                          e_fold_time_s=60.0)
    result = damp_divergence(u_div, v_div, dy_m=DY, dx_m=DX, dt_s=600.0,
                             divergent=spec, boundary="periodic",
                             periodic_domain=True)
    assert result.divergence_rms_after < 0.5 * result.divergence_rms_before
    # A purely rotational flow is untouched by the divergent-only control.
    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]
    u_rot = np.cos(2.0 * math.pi * 3 * y / ny) * np.ones((1, nx))
    v_rot = -np.cos(2.0 * math.pi * 4 * x / nx) * np.ones((ny, 1))
    untouched = damp_divergence(u_rot, v_rot, dy_m=DY, dx_m=DX, dt_s=600.0,
                                divergent=spec, boundary="periodic",
                                periodic_domain=True)
    np.testing.assert_allclose(untouched.u, u_rot, atol=1e-10)
    np.testing.assert_allclose(untouched.v, v_rot, atol=1e-10)


def test_helmholtz_decomposition_reconstructs_and_separates():
    ny, nx = 32, 40
    rng = np.random.default_rng(3)
    u = rng.normal(size=(ny, nx))
    v = rng.normal(size=(ny, nx))
    (urot, vrot), (udiv, vdiv) = helmholtz_decompose(u, v, dy_m=DY, dx_m=DX)
    np.testing.assert_allclose(urot + udiv, u, atol=1e-12)
    np.testing.assert_allclose(vrot + vdiv, v, atol=1e-12)
    # The PIN is the coefficient-space projector, and there it is exact.
    from gpuwm.spectral_ops.vector import helmholtz_coefficients
    rot, _div, _magnitude, kx, ky = helmholtz_coefficients(
        u, v, dy_m=DY, dx_m=DX)
    coefficient_divergence = kx * rot[0] + ky * rot[1]
    assert float(np.abs(coefficient_divergence).max()) < 1e-15
    # MEASURED LIMIT of the delivered arithmetic (recorded 2026-08-17, not
    # a wiring defect): realizing the parts through irfft2 loses the
    # anti-Hermitian ky content of the kx = {0, Nyquist} columns, so the
    # realized rotational field keeps a small residual divergence -- 6.1%
    # of the input divergence RMS on this white-noise operand.  The
    # damp_divergence receipt recomputes realized divergence and therefore
    # stays honest about it.  Hold the leakage under 10%.
    rot_div = spectral_divergence(urot, vrot, dy_m=DY, dx_m=DX)
    original = spectral_divergence(u, v, dy_m=DY, dx_m=DX)
    leakage = (float(np.sqrt(np.mean(rot_div ** 2)))
               / float(np.sqrt(np.mean(original ** 2))))
    assert leakage < 0.10


def test_c_grid_nonperiodic_outer_faces_receive_zero_increment():
    ny, nx = 36, 44
    rng = np.random.default_rng(5)
    u = rng.normal(size=(ny, nx + 1))
    v = rng.normal(size=(ny + 1, nx))
    spec = Hyperdiffusion(order=1, reference_wavelength_m=8.0 * DX,
                          e_fold_time_s=60.0)
    result = damp_c_grid_divergence(
        u, v, dy_m=DY, dx_m=DX, dt_s=600.0, divergent=spec,
        boundary="tapered", edge_taper_cells=6)
    np.testing.assert_array_equal(result.u[:, 0], u[:, 0])
    np.testing.assert_array_equal(result.u[:, -1], u[:, -1])
    np.testing.assert_array_equal(result.v[0, :], v[0, :])
    np.testing.assert_array_equal(result.v[-1, :], v[-1, :])
    assert result.rms_increment > 0.0


def test_c_grid_shape_contract_is_enforced():
    with pytest.raises(ValueError, match="C-grid"):
        destagger_c_grid(np.zeros((4, 4)), np.zeros((4, 4)))


def test_lift_periodic_faces_share_one_wrapped_value():
    du = np.random.default_rng(9).normal(size=(8, 8))
    dv = np.random.default_rng(10).normal(size=(8, 8))
    out_u, out_v = lift_mass_increment_to_c_grid(du, dv, periodic=True)
    np.testing.assert_array_equal(out_u[:, 0], out_u[:, -1])
    np.testing.assert_array_equal(out_v[0, :], out_v[-1, :])


def test_wind_periodic_projection_requires_the_declaration():
    u, v = _divergent_flow(16, 16)
    with pytest.raises(ValueError, match="periodic_domain"):
        damp_divergence(u, v, dy_m=DY, dx_m=DX, dt_s=60.0,
                        divergent=Hyperdiffusion(), boundary="periodic")


# ---------------------------------------------------------------------------
# elliptic solvers


def test_helmholtz_solver_matches_the_analytic_mode():
    ny, nx = 32, 32
    field = wave(ny, nx, kx=3)
    k = 2.0 * math.pi * 3 / (nx * DX)
    alpha, beta = 2.0, 1.0e7
    rhs = (alpha + beta * k * k) * field
    result = solve_helmholtz(rhs, dy_m=DY, dx_m=DX, alpha=alpha, beta=beta,
                             boundary="periodic", periodic_domain=True)
    np.testing.assert_allclose(result.values, field, atol=1e-9)
    assert result.residual_rms < 1e-9 * np.abs(rhs).max()


def test_poisson_solution_and_rhs_are_zero_mean():
    ny, nx = 24, 24
    rhs = wave(ny, nx, kx=2) + 3.0   # non-zero mean removed by policy
    result = solve_poisson(rhs, dy_m=DY, dx_m=DX, boundary="periodic",
                           periodic_domain=True)
    assert abs(float(np.mean(result.values))) < 1e-10
    assert result.rhs_mean_removed == pytest.approx(3.0, rel=1e-9)
    with pytest.raises(ValueError, match="zero horizontal mean"):
        solve_poisson(rhs, dy_m=DY, dx_m=DX, boundary="periodic",
                      periodic_domain=True, mean_policy="raise")


def test_singular_zero_mode_is_a_refusal_unless_zeroed():
    rhs = wave(16, 16, kx=1)
    with pytest.raises(ValueError, match="singular"):
        solve_helmholtz(rhs, dy_m=DY, dx_m=DX, alpha=0.0, beta=1.0,
                        boundary="periodic", periodic_domain=True,
                        zero_mode="raise")


# ---------------------------------------------------------------------------
# coupling


def test_scale_split_reconstructs_exactly():
    rng = np.random.default_rng(13)
    field = rng.normal(size=(32, 48))
    low_pass = RaisedCosineLowPass(pass_wavelength_m=24.0 * DX,
                                   stop_wavelength_m=8.0 * DX)
    split = split_scales(field, dy_m=DY, dx_m=DX, low_pass=low_pass,
                         boundary="reflect")
    np.testing.assert_allclose(split.low + split.high, field, atol=1e-12)
    assert split.reconstruction_max_abs < 1e-12


def test_blend_refuses_mismatched_grids():
    low_pass = RaisedCosineLowPass(pass_wavelength_m=24.0 * DX,
                                   stop_wavelength_m=8.0 * DX)
    with pytest.raises(ValueError, match="common grid"):
        blend_parent_child(np.zeros((8, 8)), np.zeros((8, 9)),
                           dy_m=DY, dx_m=DX, low_pass=low_pass)


def test_tapered_nudging_is_weak_bounded_and_zero_at_the_boundary():
    rng = np.random.default_rng(17)
    current = rng.normal(size=(40, 40))
    target = current + rng.normal(size=(40, 40))
    low_pass = RaisedCosineLowPass(pass_wavelength_m=20.0 * DX,
                                   stop_wavelength_m=10.0 * DX)
    result = nudge_large_scales(
        current, target, dy_m=DY, dx_m=DX, dt_s=60.0,
        relaxation_time_s=3600.0, low_pass=low_pass, boundary="tapered",
        edge_taper_cells=6)
    assert 0.0 < result.relaxation_fraction < 0.02
    # The tapered filter preserves its OPERAND (the mismatch) at the outer
    # edge, so the edge increment is the full relaxed raw mismatch there --
    # the receipt reports it as boundary_max_abs instead of claiming zero.
    limit = result.relaxation_fraction * float(
        np.abs(target - current).max()) + 1e-12
    assert 0.0 < result.boundary_max_abs <= limit
    assert result.increment_max_abs <= float(np.abs(target - current).max())


# ---------------------------------------------------------------------------
# adaptive calibration


def test_calibration_damps_the_observed_excess_and_never_amplifies():
    observations = [
        BandObservation(wavelength_m=24_000.0, power_ratio=1.05),
        BandObservation(wavelength_m=12_000.0, power_ratio=1.8),
        BandObservation(wavelength_m=6_000.0, power_ratio=4.0),
    ]
    result = fit_hyperdiffusion(observations, dt_s=60.0)
    assert result["status"] == "proposal-only"
    predicted = np.asarray(result["predicted_amplitude_transfer"])
    assert np.all(predicted <= 1.0 + 1e-12)
    assert predicted[-1] < predicted[0]          # strongest damping smallest
    body = dict(result)
    digest = body.pop("recommendation_sha256")
    assert canonical_hash(body) == digest


def test_calibration_is_deterministic():
    observations = [BandObservation(wavelength_m=9_000.0, power_ratio=2.0)]
    one = fit_hyperdiffusion(observations, dt_s=30.0)
    two = fit_hyperdiffusion(observations, dt_s=30.0)
    assert one == two


# ---------------------------------------------------------------------------
# runtime contracts


def shadow_config(**overrides):
    spec = Hyperdiffusion(order=3, reference_wavelength_m=6.0 * DX,
                          e_fold_time_s=300.0)
    values = dict(
        mode="shadow", boundary="tapered", edge_taper_cells=6,
        scalar_targets=(ScalarTarget(field="thp", diffusion=spec),))
    values.update(overrides)
    config = SpectralNumericsConfig(**values)
    config.validate()
    return config


def test_off_mode_reads_no_state_and_returns_none():
    config = SpectralNumericsConfig()          # mode="off"
    result = apply_spectral_step(
        ReadsNothing(), config=config, dx_m=DX, dy_m=DY, dt_s=60.0, step=1)
    assert result is None


def test_non_cadence_steps_read_no_state_and_return_none():
    config = shadow_config(cadence_steps=4)
    result = apply_spectral_step(
        ReadsNothing(), config=config, dx_m=DX, dy_m=DY, dt_s=60.0, step=3)
    assert result is None


def test_shadow_mode_is_state_bitwise_inert_and_makes_a_receipt():
    rng = np.random.default_rng(23)
    field = rng.normal(size=(3, 32, 32))
    state = {"thp": field.copy()}
    receipt = apply_spectral_step(
        state, config=shadow_config(), dx_m=DX, dy_m=DY, dt_s=60.0, step=1,
        source={"domain": "d01"})
    np.testing.assert_array_equal(state["thp"], field)
    assert receipt["mode"] == "shadow"
    assert receipt["applied"] is False
    assert receipt["operator_pins_sha256"] == COMMITTED_OPERATOR_PINS_SHA256
    assert receipt["metrics"]["scalar"]["thp"]["rms_increment"] > 0.0
    validate_receipt(receipt)


def test_apply_mode_mutates_only_the_configured_fields():
    rng = np.random.default_rng(29)
    field = rng.normal(size=(3, 32, 32))
    other = rng.normal(size=(3, 32, 32))
    state = {"thp": field.copy(), "qv": other.copy()}
    receipt = apply_spectral_step(
        state, config=shadow_config(mode="apply"), dx_m=DX, dy_m=DY,
        dt_s=60.0, step=1)
    assert receipt["applied"] is True
    assert not np.array_equal(state["thp"], field)
    np.testing.assert_array_equal(state["qv"], other)


def test_budget_violation_raises_before_any_mutation():
    rng = np.random.default_rng(31)
    field = rng.normal(size=(3, 32, 32))
    state = {"thp": field.copy()}
    config = shadow_config(mode="apply",
                           maximum_scalar_rms_increment=1.0e-30)
    with pytest.raises(SpectralBudgetExceeded):
        apply_spectral_step(state, config=config, dx_m=DX, dy_m=DY,
                            dt_s=60.0, step=1)
    np.testing.assert_array_equal(state["thp"], field)


def test_missing_target_field_is_a_refusal():
    with pytest.raises(ValueError, match="missing spectral target"):
        apply_spectral_step({}, config=shadow_config(), dx_m=DX, dy_m=DY,
                            dt_s=60.0, step=1)


def test_active_mode_with_no_targets_is_a_refusal():
    config = SpectralNumericsConfig(mode="shadow")
    with pytest.raises(ValueError, match="no scalar or wind target"):
        apply_spectral_step({}, config=config, dx_m=DX, dy_m=DY, dt_s=60.0,
                            step=1)


def test_receipts_are_written_atomically_when_a_directory_is_configured(
        tmp_path):
    state = {"thp": np.random.default_rng(37).normal(size=(2, 24, 24))}
    config = shadow_config(receipt_directory=str(tmp_path))
    receipt = apply_spectral_step(state, config=config, dx_m=DX, dy_m=DY,
                                  dt_s=60.0, step=7)
    path = tmp_path / "spectral-step-000000007.json"
    assert path.is_file()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(receipt))
    validate_receipt(on_disk)


def test_wind_control_through_the_runtime_step():
    ny, nx = 32, 32
    rng = np.random.default_rng(41)
    state = {"U": rng.normal(size=(2, ny, nx + 1)),
             "V": rng.normal(size=(2, ny + 1, nx))}
    u0 = state["U"].copy()
    v0 = state["V"].copy()
    config = SpectralNumericsConfig(
        mode="apply", boundary="tapered", edge_taper_cells=6,
        wind=WindControl(enabled=True))
    config.validate()
    receipt = apply_spectral_step(state, config=config, dx_m=DX, dy_m=DY,
                                  dt_s=60.0, step=1)
    assert receipt["metrics"]["wind"]["staggering"] == "cgrid"
    # The outer faces carry externally supplied boundary values: untouched.
    np.testing.assert_array_equal(state["U"][..., :, 0], u0[..., :, 0])
    np.testing.assert_array_equal(state["U"][..., :, -1], u0[..., :, -1])
    np.testing.assert_array_equal(state["V"][..., 0, :], v0[..., 0, :])
    np.testing.assert_array_equal(state["V"][..., -1, :], v0[..., -1, :])
    assert not np.array_equal(state["U"], u0)


# ---------------------------------------------------------------------------
# receipts


def test_receipt_hash_binds_the_contents():
    receipt = make_receipt(
        config_sha256="0" * 64, mode="shadow", step=3, dt_s=60.0, dx_m=DX,
        dy_m=DY, backend="numpy", metrics={"scalar": {}, "wind": None},
        applied=False)
    assert receipt["schema"] == RECEIPT_SCHEMA
    validate_receipt(receipt)
    tampered = dict(receipt)
    tampered["step"] = 4
    with pytest.raises(ValueError, match="hash"):
        validate_receipt(tampered)


def test_receipt_from_another_operator_identity_is_refused():
    receipt = make_receipt(
        config_sha256="0" * 64, mode="shadow", step=1, dt_s=60.0, dx_m=DX,
        dy_m=DY, backend="numpy", metrics={}, applied=False)
    foreign = dict(receipt)
    foreign["operator_pins_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="operator identity"):
        validate_receipt(foreign)


# ---------------------------------------------------------------------------
# configuration


def test_from_mapping_none_is_the_off_default():
    config = from_mapping(None)
    assert config.mode == "off"
    assert config.scalar_targets == ()
    assert not config.wind.enabled


def test_from_mapping_builds_scalar_and_wind_tables():
    config = from_mapping({
        "mode": "shadow",
        "cadence_steps": 3,
        "boundary": "tapered",
        "edge_taper_cells": 8,
        "scalar": [{"field": "thp",
                    "diffusion": {"order": 3,
                                  "reference_wavelength_m": 18_000.0,
                                  "e_fold_time_s": 450.0}}],
        "wind": {"enabled": True,
                 "divergent": {"order": 2,
                               "reference_wavelength_m": 12_000.0,
                               "e_fold_time_s": 300.0}},
    })
    assert config.mode == "shadow"
    assert config.cadence_steps == 3
    assert config.scalar_targets[0].field == "thp"
    assert config.wind.enabled
    assert len(config.config_sha256) == 64


def test_from_mapping_refuses_unknown_keys_everywhere():
    with pytest.raises(ValueError, match="unknown spectral_numerics keys"):
        from_mapping({"mode": "shadow", "cadence": 2})
    with pytest.raises(ValueError, match="unknown hyperdiffusion keys"):
        from_mapping({"mode": "shadow",
                      "scalar": [{"field": "thp",
                                  "diffusion": {"tau_s": 300.0}}]})


def test_config_validation_refusals_name_their_breakage():
    with pytest.raises(ValueError, match="periodic_domain"):
        from_mapping({"mode": "shadow", "boundary": "periodic",
                      "scalar": [{"field": "thp", "diffusion": {}}]})
    with pytest.raises(ValueError, match="reflect is scalar-only"):
        from_mapping({"mode": "shadow", "boundary": "reflect",
                      "wind": {"enabled": True}})
    with pytest.raises(ValueError, match="unique"):
        from_mapping({"mode": "shadow",
                      "scalar": [{"field": "thp", "diffusion": {}},
                                 {"field": "thp", "diffusion": {}}]})


def test_config_hash_is_stable_across_equal_documents():
    mapping = {"mode": "shadow",
               "scalar": [{"field": "thp", "diffusion": {}}]}
    assert (from_mapping(mapping).config_sha256
            == from_mapping(json.loads(json.dumps(mapping))).config_sha256)


# ---------------------------------------------------------------------------
# benchmark controls


def test_benchmark_controls_hold_on_a_small_cpu_case():
    from gpuwm.spectral_ops.benchmark import run_benchmark
    result = run_benchmark(nx=48, ny=32, levels=2, repeats=1,
                           backend="numpy")
    controls = result["controls"]
    assert controls["divergence_ratio"] < 1.0
    assert abs(controls["scalar_mean_drift"]) < 1e-6
    assert controls["helmholtz_residual_rms"] < 1e-4


def test_two_domains_writing_the_same_step_do_not_overwrite_each_other(
        tmp_path):
    """A nested chain commits step N once per domain, so a receipt file
    named by the step alone is written twice and only the last one
    survives.  MEASURED on a real two-domain HRRR shadow run
    (2026-08-18): 300 receipts were computed -- 60 on d01, 240 on d02 --
    and 240 files existed afterwards, 59 of them d01's.  The capsule
    ledger was right and the on-disk audit trail was short by 60, which
    for an apply run is exactly the state ``require_complete`` exists to
    forbid: a state change whose receipt is gone.
    """
    rng = np.random.default_rng(101)
    config = shadow_config(receipt_directory=str(tmp_path))
    written = []
    for domain, grid_id in (("d01", 1), ("d02", 2)):
        state = {"thp": rng.normal(size=(2, 24, 24))}
        written.append(apply_spectral_step(
            state, config=config, dx_m=DX, dy_m=DY, dt_s=60.0, step=7,
            source={"domain": domain, "grid_id": grid_id}))
    files = sorted(path.name for path in tmp_path.glob("*.json"))
    assert len(files) == 2, (
        "each domain's step-7 receipt needs its own file; these "
        f"collided: {files}")
    on_disk = [json.loads((tmp_path / name).read_text(encoding="utf-8"))
               for name in files]
    assert {row["source"]["domain"] for row in on_disk} == {"d01", "d02"}
    for row in on_disk:
        validate_receipt(row)
    assert on_disk == sorted(
        (json.loads(json.dumps(row)) for row in written),
        key=lambda row: row["source"]["domain"])


def test_a_receipt_with_no_domain_keeps_the_bare_step_name(tmp_path):
    """``apply_spectral_step`` is callable without a source identity --
    the CLI and the delivered demo receipt both are -- and that spelling
    must not move."""
    state = {"thp": np.random.default_rng(37).normal(size=(2, 24, 24))}
    config = shadow_config(receipt_directory=str(tmp_path))
    apply_spectral_step(state, config=config, dx_m=DX, dy_m=DY, dt_s=60.0,
                        step=7)
    assert (tmp_path / "spectral-step-000000007.json").is_file()


def test_a_receipt_file_is_the_same_bytes_on_every_platform(tmp_path):
    """``Path.write_text`` without an explicit newline translates "\n" to
    the platform's line ending, so the identical receipt landed as LF on
    Linux and CRLF on Windows (measured: the two receipts pulled into
    this lane's evidence arrived CRLF).  A receipt whose bytes depend on
    the operating system cannot be compared across a dual run or carried
    between a node and this box, so the writer pins LF.
    """
    state = {"thp": np.random.default_rng(53).normal(size=(2, 24, 24))}
    config = shadow_config(receipt_directory=str(tmp_path))
    apply_spectral_step(state, config=config, dx_m=DX, dy_m=DY, dt_s=60.0,
                        step=11, source={"domain": "d01"})
    raw = (tmp_path / "d01-spectral-step-000000011.json").read_bytes()
    assert b"\r" not in raw, "spectral receipts are written with LF endings"
    validate_receipt(json.loads(raw.decode("utf-8")))
