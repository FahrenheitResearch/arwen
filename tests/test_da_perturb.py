"""Tests for gpuwm.da.perturb.

These are written to falsify the module's four claims rather than to exercise
it: the field really has the prescribed length scale (measured off its own
power spectrum), the draw really is reproducible from the seed alone, the
bounds really hold on states chosen to break them, and the rim taper really
is exactly zero on the boundary and exactly one inside.

CuPy is imported inside test functions only.  ``tests/conftest.py``
AST-scans each module and marks any module that imports cupy at top level as
``gpu``, which would exclude this whole file from ``-m "not gpu"``.
"""

import json
import math

import numpy as np
import pytest

from conftest import requires_gpu

from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.state import DomainState
from gpuwm.da import STATUS
from gpuwm.da.perturb import (
    FieldPerturbation,
    PerturbationConfig,
    PROVENANCE_SCHEMA,
    SUPPORTED_FIELDS,
    apply_perturbations,
    boundary_taper,
    fit_gaussian_length_scale,
    gaussian_random_field,
    perturbed_lateral_boundaries,
    radial_power_spectrum,
    recycled_difference_perturbations,
    spectral_peak_wavenumber,
)

#: The spectral tests use a domain big enough that the prescribed length
#: scale sits ~10 fundamental wavenumbers away from k=0 and far from Nyquist.
#: At 4 km on a 256 km domain, 1/L = 0.25 rad/km and the fundamental is
#: 0.0245 rad/km.  Shrink the domain and the peak walks into the first bin,
#: where there are too few discrete modes to locate it.
SPECTRAL_SHAPE = (24, 256, 256)
SPECTRAL_DX_KM = 1.0
SPECTRAL_L_KM = 4.0
SPECTRAL_BINS = 48

SEEDS = (1, 2, 3, 17, 999, 4242, 777)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _cfg(nx=48, ny=40, nz=12, **kwargs):
    kwargs.setdefault("moist", True)
    return RunConfig(nx=nx, ny=ny, nz=nz, dx=1000.0, dy=1000.0, ztop=6400.0,
                     dt=1.0, run_seconds=1.0, **kwargs)


def _prepared_state(nx=48, ny=40, nz=12, *, qv=0.0, theta_base=300.0,
                    pressure=None):
    """A host DomainState with a usable pressure, base theta, and moisture.

    A freshly constructed state has ``p`` all zeros, which the module refuses
    to divide by; filling it here is what the real init path would have done.
    """
    state = DomainState(_cfg(nx, ny, nz), array_module=np)
    if pressure is None:
        column = np.linspace(1.0e5, 2.0e4, nz, dtype=np.float32)
        pressure = np.broadcast_to(column[:, None, None], (nz, ny, nx))
    state.p[...] = np.asarray(pressure, dtype=np.float32)
    state.thb[...] = np.float32(theta_base)
    state.qv[...] = np.float32(0.0) + qv
    return state


def _tetens_qvs(temperature, pressure):
    """Saturation mixing ratio, restated independently of the module."""
    es = 1000.0 * c.SVP1 * np.exp(
        c.SVP2 * (temperature - c.SVPT0) / (temperature - c.SVP3))
    return c.EP2 * es / (pressure - es)


def _state_temperature(state):
    theta = state.thb[:, None, None] + state.thp
    return theta * (state.p / c.P0) ** c.RCP


def _spec(name, amplitude, length_scale_km=4.0, vertical_scale_levels=0.0):
    return FieldPerturbation(name=name, amplitude=amplitude,
                             length_scale_km=length_scale_km,
                             vertical_scale_levels=vertical_scale_levels)


def _pcfg(*specs, **kwargs):
    kwargs.setdefault("dx_km", 1.0)
    kwargs.setdefault("dy_km", 1.0)
    # ``_prepared_state`` has no vertical coordinate (dnw is all zero), so
    # the production default, mass_balance = "hydrostatic", refuses it by
    # name (see test_mass_balance_refuses_a_state_without_a_vertical_
    # coordinate).  The draw mechanics are tested with the balance off;
    # the balance itself is tested on ``_column_state`` below.
    kwargs.setdefault("mass_balance", "none")
    return PerturbationConfig(fields=tuple(specs), **kwargs)


# --------------------------------------------------------------------------
# 1. the spectrum peaks where the length scale says it should
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_radial_spectrum_peaks_at_prescribed_length_scale(seed):
    """``E(k)`` peaks at ``k = 1/L``, within a 15% band.

    For a Gaussian correlation ``exp(-r^2/2L^2)`` the radial spectrum is
    ``E(k) ~ k exp(-k^2 L^2 / 2)``, whose only stationary point is
    ``k = 1/L``.  Measured over seven seeds the realized ratio ``k_peak * L``
    spans 0.99 to 1.09; the band here is 0.85-1.15 so the test reports a
    broken length scale, not a broken random number.
    """
    field, _ = gaussian_random_field(
        SPECTRAL_SHAPE, seed=seed, name="t", dx_km=SPECTRAL_DX_KM,
        dy_km=SPECTRAL_DX_KM, length_scale_km=SPECTRAL_L_KM,
        vertical_scale_levels=0.0, xp=np)
    k, energy = radial_power_spectrum(field, SPECTRAL_DX_KM,
                                      bins=SPECTRAL_BINS)
    peak = spectral_peak_wavenumber(k, energy)
    ratio = peak * SPECTRAL_L_KM
    assert 0.85 <= ratio <= 1.15, (
        f"seed {seed}: spectral peak at k={peak:.5f} rad/km, expected "
        f"{1.0 / SPECTRAL_L_KM:.5f} (ratio {ratio:.4f})")


@pytest.mark.parametrize("seed", SEEDS)
def test_spectral_shape_recovers_the_length_scale_to_two_percent(seed):
    """The whole spectral shape, not just its peak, is the prescribed one.

    ``log(E/k)`` is exactly linear in ``k^2`` with slope ``-L^2/2`` for this
    construction.  Regressing it back recovers L to better than 1% over
    seven seeds; the gate is 2%.  This is the sharp version of the peak test
    above -- a field with the right peak but the wrong fall-off fails here.
    """
    field, _ = gaussian_random_field(
        SPECTRAL_SHAPE, seed=seed, name="t", dx_km=SPECTRAL_DX_KM,
        dy_km=SPECTRAL_DX_KM, length_scale_km=SPECTRAL_L_KM,
        vertical_scale_levels=0.0, xp=np)
    k, energy = radial_power_spectrum(field, SPECTRAL_DX_KM,
                                      bins=SPECTRAL_BINS)
    recovered = fit_gaussian_length_scale(k, energy)
    assert recovered == pytest.approx(SPECTRAL_L_KM, rel=0.02), (
        f"seed {seed}: recovered L={recovered:.4f} km from the spectrum, "
        f"prescribed {SPECTRAL_L_KM} km")


@pytest.mark.parametrize("length_km", [3.0, 5.0, 8.0])
def test_the_peak_moves_with_the_prescribed_scale(length_km):
    """Doubling the length scale halves the peak wavenumber.

    Guards against a spectrum that happens to peak in the right place for
    one configuration because of the binning rather than the physics.
    """
    field, _ = gaussian_random_field(
        SPECTRAL_SHAPE, seed=31, name="t", dx_km=SPECTRAL_DX_KM,
        dy_km=SPECTRAL_DX_KM, length_scale_km=length_km,
        vertical_scale_levels=0.0, xp=np)
    k, energy = radial_power_spectrum(field, SPECTRAL_DX_KM,
                                      bins=SPECTRAL_BINS)
    assert fit_gaussian_length_scale(k, energy) == pytest.approx(
        length_km, rel=0.03)


def test_anisotropic_grid_spacing_is_honoured():
    """A length scale in km is a length scale in km on both axes.

    With ``dy = 2 dx`` the same physical L must come back, which it only
    does if the filter is built from the two spacings separately rather
    than from a single assumed one.
    """
    field, _ = gaussian_random_field(
        (16, 128, 256), seed=5, name="t", dx_km=1.0, dy_km=2.0,
        length_scale_km=6.0, vertical_scale_levels=0.0, xp=np)
    k, energy = radial_power_spectrum(field, 1.0, 2.0, bins=32)
    assert fit_gaussian_length_scale(k, energy) == pytest.approx(6.0,
                                                                 rel=0.05)


def test_amplitude_is_normalized_to_unit_variance():
    """The analytic normalization puts the realized RMS within a few percent
    of 1 without dividing out the sample standard deviation.

    Exactly 1.0 would be the bug: forcing every member to the same realized
    variance removes the sampling fluctuation an ensemble is supposed to
    have.
    """
    realized = []
    for seed in SEEDS:
        _, info = gaussian_random_field(
            (16, 128, 128), seed=seed, name="t", dx_km=1.0, dy_km=1.0,
            length_scale_km=4.0, vertical_scale_levels=0.0, xp=np)
        realized.append(info["realized_rms"])
    assert all(0.9 < value < 1.1 for value in realized), realized
    assert len(set(realized)) == len(realized), (
        "every seed produced the identical realized RMS, which means the "
        "normalization is dividing by the sample standard deviation")


# --------------------------------------------------------------------------
# 2. vertical correlation
# --------------------------------------------------------------------------

def _lag1_level_correlation(field):
    flat = field.reshape(field.shape[0], -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    numerator = (flat[:-1] * flat[1:]).mean(axis=1)
    denominator = np.sqrt((flat[:-1] ** 2).mean(axis=1)
                          * (flat[1:] ** 2).mean(axis=1))
    return float(np.mean(numerator / denominator))


def test_zero_vertical_scale_decorrelates_the_levels():
    field, _ = gaussian_random_field(
        (24, 96, 96), seed=11, name="t", dx_km=1.0, dy_km=1.0,
        length_scale_km=4.0, vertical_scale_levels=0.0, xp=np)
    assert abs(_lag1_level_correlation(field)) < 0.05


def test_vertical_scale_correlates_adjacent_levels():
    """``exp(-1/(2 Lv^2))`` at lag 1, so Lv=6 levels means ~0.99."""
    field, _ = gaussian_random_field(
        (24, 96, 96), seed=11, name="t", dx_km=1.0, dy_km=1.0,
        length_scale_km=4.0, vertical_scale_levels=6.0, xp=np)
    correlation = _lag1_level_correlation(field)
    expected = math.exp(-1.0 / (2.0 * 6.0 ** 2))
    assert correlation == pytest.approx(expected, abs=0.05), correlation


# --------------------------------------------------------------------------
# 3. determinism
# --------------------------------------------------------------------------

def test_same_seed_field_and_shape_reproduce_bit_for_bit():
    first, info_a = gaussian_random_field(
        (6, 32, 32), seed=1234, name="qv", dx_km=1.0, dy_km=1.0,
        length_scale_km=4.0, xp=np)
    second, info_b = gaussian_random_field(
        (6, 32, 32), seed=1234, name="qv", dx_km=1.0, dy_km=1.0,
        length_scale_km=4.0, xp=np)
    assert np.array_equal(first, second)
    assert info_a["noise_sha256"] == info_b["noise_sha256"]
    assert info_a["stream_key_hex"] == info_b["stream_key_hex"]


@pytest.mark.parametrize("changed", ["seed", "name", "shape"])
def test_changing_any_of_seed_field_or_shape_changes_the_draw(changed):
    base = dict(shape=(6, 32, 32), seed=1234, name="qv", dx_km=1.0,
                dy_km=1.0, length_scale_km=4.0, xp=np)
    other = dict(base)
    if changed == "seed":
        other["seed"] = 1235
    elif changed == "name":
        other["name"] = "t"
    else:
        other["shape"] = (6, 32, 33)
    _, info_a = gaussian_random_field(**base)
    _, info_b = gaussian_random_field(**other)
    assert info_a["noise_sha256"] != info_b["noise_sha256"]
    assert info_a["stream_key_hex"] != info_b["stream_key_hex"]


def test_adjacent_seeds_are_not_adjacent_streams():
    """Hash-derived keys, not seed arithmetic: member 1 and member 2 must
    not be correlated draws of one stream."""
    fields = []
    for seed in (1, 2):
        field, _ = gaussian_random_field(
            (8, 64, 64), seed=seed, name="t", dx_km=1.0, dy_km=1.0,
            length_scale_km=4.0, xp=np)
        fields.append(field.ravel())
    correlation = float(np.corrcoef(fields[0], fields[1])[0, 1])
    assert abs(correlation) < 0.05, correlation


def test_apply_perturbations_is_reproducible_from_the_seed():
    specs = (_spec("theta", 1.0), _spec("qv", 5.0e-4), _spec("u", 1.5),
             _spec("v", 1.5))
    states = []
    provenances = []
    for _ in range(2):
        state = _prepared_state(qv=1.0e-3)
        provenances.append(apply_perturbations(state, 20260730, _pcfg(*specs)))
        states.append(state)
    for name in ("thp", "qv", "u", "v"):
        assert np.array_equal(getattr(states[0], name),
                              getattr(states[1], name)), name
    assert provenances[0] == provenances[1]


def test_two_members_actually_differ():
    """The point of the module: different seeds give different states."""
    specs = (_spec("theta", 1.0),)
    first = _prepared_state()
    second = _prepared_state()
    apply_perturbations(first, 1, _pcfg(*specs))
    apply_perturbations(second, 2, _pcfg(*specs))
    assert not np.array_equal(first.thp, second.thp)
    spread = float(np.std(first.thp - second.thp))
    assert spread > 0.5, spread


# --------------------------------------------------------------------------
# 4. the rim taper
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["cosine", "linear"])
@pytest.mark.parametrize("rim", [1, 3, 5])
def test_taper_is_exactly_zero_on_the_boundary_and_exactly_one_inside(kind,
                                                                     rim):
    taper = boundary_taper(31, 37, rim, kind=kind)
    assert taper.shape == (31, 37)
    for edge in (taper[0, :], taper[-1, :], taper[:, 0], taper[:, -1]):
        assert np.all(edge == 0.0), "boundary is not exactly zero"
    interior = taper[rim:31 - rim, rim:37 - rim]
    assert np.all(interior == 1.0), "interior is not exactly one"
    assert float(taper.min()) == 0.0
    assert float(taper.max()) == 1.0


@pytest.mark.parametrize("kind", ["cosine", "linear"])
def test_taper_is_monotone_from_the_edge_inward(kind):
    taper = boundary_taper(41, 41, 6, kind=kind)
    profile = taper[20, :21]
    assert np.all(np.diff(profile) >= 0.0)
    assert profile[0] == 0.0
    assert profile[6] == 1.0


def test_taper_rejects_a_rim_that_swallows_the_domain():
    with pytest.raises(ValueError, match="no untapered interior"):
        boundary_taper(10, 10, 5)


def test_taper_rejects_a_zero_rim():
    with pytest.raises(ValueError, match="rim_width must be >= 1"):
        boundary_taper(40, 40, 0)


def test_state_boundary_rows_are_untouched_bit_for_bit():
    """The boundary claim, stated on the state rather than on the taper.

    Members share one boundary file; if a single boundary value moved, the
    member's first step would relax toward a value it does not hold.
    """
    state = _prepared_state(qv=1.0e-3)
    state.thp[...] = np.float32(0.25)
    state.u[...] = np.float32(7.0)
    state.v[...] = np.float32(-3.0)
    before = {name: getattr(state, name).copy()
              for name in ("thp", "qv", "u", "v")}
    apply_perturbations(state, 5, _pcfg(
        _spec("theta", 2.0), _spec("qv", 1.0e-3), _spec("u", 3.0),
        _spec("v", 3.0)))
    for name in ("thp", "qv", "u", "v"):
        now, was = getattr(state, name), before[name]
        assert np.array_equal(now[:, 0, :], was[:, 0, :]), f"{name} south"
        assert np.array_equal(now[:, -1, :], was[:, -1, :]), f"{name} north"
        assert np.array_equal(now[:, :, 0], was[:, :, 0]), f"{name} west"
        assert np.array_equal(now[:, :, -1], was[:, :, -1]), f"{name} east"
        assert not np.array_equal(now, was), f"{name} was not perturbed at all"


# --------------------------------------------------------------------------
# 5. staggering
# --------------------------------------------------------------------------

def test_staggered_shapes_are_preserved_and_all_are_perturbed():
    state = _prepared_state(nx=48, ny=40, nz=12, qv=1.0e-3)
    provenance = apply_perturbations(state, 9, _pcfg(
        _spec("theta", 1.0), _spec("qv", 1.0e-4), _spec("u", 1.0),
        _spec("v", 1.0)))
    assert state.thp.shape == (12, 40, 48)
    assert state.qv.shape == (12, 40, 48)
    assert state.u.shape == (12, 40, 49)
    assert state.v.shape == (12, 41, 48)
    shapes = {record["name"]: tuple(record["shape"])
              for record in provenance["fields"]}
    assert shapes == {"theta": (12, 40, 48), "qv": (12, 40, 48),
                      "u": (12, 40, 49), "v": (12, 41, 48)}
    for name in ("thp", "qv", "u", "v"):
        assert np.count_nonzero(getattr(state, name)) > 0


def test_a_mis_staggered_state_is_rejected_not_broadcast():
    state = _prepared_state()
    state.u = np.zeros((12, 40, 48), dtype=np.float32)  # mass-shaped u
    with pytest.raises(ValueError, match="ARW staggering"):
        apply_perturbations(state, 1, _pcfg(_spec("u", 1.0),
                                            wind_mode="independent"))


def test_dtype_and_backend_survive_the_perturbation():
    state = _prepared_state(qv=1.0e-3)
    apply_perturbations(state, 1, _pcfg(_spec("theta", 1.0),
                                        _spec("u", 1.0),
                                        wind_mode="independent"))
    assert state.thp.dtype == np.float32
    assert state.u.dtype == np.float32
    assert isinstance(state.thp, np.ndarray)


# --------------------------------------------------------------------------
# 6. bounds on adversarial states
# --------------------------------------------------------------------------

def test_near_zero_moisture_never_goes_negative():
    """Adversarial: qv is 1e-8 everywhere and the perturbation is 1e-3.

    Without the floor roughly half the domain would go negative; the
    assertion on the clip count is there so a future change that stops the
    perturbation reaching qv cannot make this test pass vacuously.
    """
    state = _prepared_state(qv=1.0e-8)
    provenance = apply_perturbations(
        state, 77, _pcfg(_spec("qv", 1.0e-3), rh_cap=None))
    assert float(state.qv.min()) >= 0.0
    assert provenance["bounds"]["qv_floor_clipped_points"] > 0


def test_a_nonzero_moisture_floor_is_honoured():
    """The floor is applied in the field's own dtype.

    ``float32(1e-9)`` is 9.9999997e-10, a hair under the configured value;
    clamping to the float64 number would round back below it on store.  The
    bound that holds is the representable one, and the test says so rather
    than pretending otherwise.
    """
    state = _prepared_state(qv=1.0e-8)
    provenance = apply_perturbations(
        state, 77, _pcfg(_spec("qv", 1.0e-3), qv_floor=1.0e-9, rh_cap=None))
    assert float(state.qv.min()) >= float(np.float32(1.0e-9))
    assert provenance["bounds"]["qv_floor"] == 1.0e-9
    assert provenance["bounds"]["qv_floor_clipped_points"] > 0


def test_near_saturated_state_is_not_pushed_past_the_cap():
    """Adversarial: qv starts at 99.5% RH everywhere.

    The cap is checked against a saturation mixing ratio computed here from
    the Tetens formula rather than from the module, so an error in the
    module's own saturation curve cannot hide.
    """
    nz, ny, nx = 12, 40, 48
    state = _prepared_state(nx=nx, ny=ny, nz=nz)
    qvs = _tetens_qvs(_state_temperature(state), state.p)
    state.qv[...] = (0.995 * qvs).astype(np.float32)
    provenance = apply_perturbations(
        state, 8, _pcfg(_spec("qv", 2.0e-3), rh_cap=1.0))
    assert provenance["bounds"]["rh_cap_clipped_points"] > 0
    ceiling = _tetens_qvs(_state_temperature(state), state.p)
    # float32 storage of a float64 ceiling: allow one part in 1e-5.
    assert np.all(state.qv <= ceiling * (1.0 + 1.0e-5))


def test_a_sub_unity_cap_is_honoured():
    state = _prepared_state(nx=48, ny=40, nz=12)
    qvs = _tetens_qvs(_state_temperature(state), state.p)
    state.qv[...] = (0.8 * qvs).astype(np.float32)
    apply_perturbations(state, 8, _pcfg(_spec("qv", 2.0e-3), rh_cap=0.9))
    ceiling = 0.9 * _tetens_qvs(_state_temperature(state), state.p)
    assert np.all(state.qv <= ceiling * (1.0 + 1.0e-5))


def test_the_cap_uses_the_perturbed_temperature():
    """Warming the column raises the ceiling; the cap must see that.

    A cap evaluated against the unperturbed theta would clip a saturated
    column that the warming had just un-saturated.
    """
    state = _prepared_state(nx=48, ny=40, nz=12)
    qvs = _tetens_qvs(_state_temperature(state), state.p)
    state.qv[...] = (0.999 * qvs).astype(np.float32)
    apply_perturbations(state, 3, _pcfg(_spec("theta", 3.0),
                                        _spec("qv", 1.0e-5), rh_cap=1.0))
    warmed_ceiling = _tetens_qvs(_state_temperature(state), state.p)
    assert np.all(state.qv <= warmed_ceiling * (1.0 + 1.0e-5))
    # Somewhere the column warmed enough that the pre-perturbation ceiling
    # is now exceeded -- proving the ceiling really moved with the state.
    assert np.any(state.qv > qvs)


def test_the_cap_does_not_repair_the_untapered_rim():
    """A rim that arrives supersaturated leaves supersaturated.

    The rim must match the shared boundary file byte for byte.  A clamp that
    "helpfully" fixed it would produce a member whose boundary row disagrees
    with the forcing it is about to be relaxed toward -- which is the exact
    failure the taper exists to prevent.  This was a real defect: the first
    version of the cap clipped the whole array.
    """
    state = _prepared_state(nx=48, ny=40, nz=12)
    qvs = _tetens_qvs(_state_temperature(state), state.p)
    # A uniform 1e-3 is wildly supersaturated at the top of this column and
    # subsaturated at the bottom: violations exist on the rim before the
    # module is called at all.
    state.qv[...] = np.float32(1.0e-3)
    assert np.any(state.qv[:, 0, :] > qvs[:, 0, :]), "test setup is inert"
    before = state.qv.copy()

    provenance = apply_perturbations(
        state, 12, _pcfg(_spec("qv", 2.0e-4), rh_cap=1.0))

    assert np.array_equal(state.qv[:, 0, :], before[:, 0, :])
    assert np.array_equal(state.qv[:, -1, :], before[:, -1, :])
    assert np.array_equal(state.qv[:, :, 0], before[:, :, 0])
    assert np.array_equal(state.qv[:, :, -1], before[:, :, -1])
    # It did clamp in the interior, and it said the violation predated it.
    assert provenance["bounds"]["rh_cap_clipped_points"] > 0
    assert provenance["bounds"]["pre_existing_supersaturated_points"] > 0


def test_bounds_report_says_it_skipped_when_moisture_is_untouched():
    state = _prepared_state(qv=1.0e-3)
    provenance = apply_perturbations(state, 1, _pcfg(_spec("theta", 1.0)))
    assert provenance["bounds"]["evaluated"] is False
    assert "no moisture perturbation" in provenance["bounds"]["skipped_reason"]


# --------------------------------------------------------------------------
# 7. temperature vs potential temperature
# --------------------------------------------------------------------------

def test_theta_amplitude_is_applied_directly():
    """Exact reconstruction of the applied increment, staggering included."""
    state = _prepared_state(nx=48, ny=40, nz=12)
    cfg = _pcfg(_spec("theta", 2.0))
    apply_perturbations(state, 42, cfg)
    draw, _ = gaussian_random_field(
        (12, 40, 48), seed=42, name="theta", dx_km=1.0, dy_km=1.0,
        length_scale_km=4.0, vertical_scale_levels=0.0, xp=np)
    taper = boundary_taper(40, 48, 5, kind="cosine")
    expected = (2.0 * draw * taper[None, :, :]).astype(np.float32)
    assert np.allclose(state.thp, expected, rtol=0.0, atol=1e-6)


def test_temperature_amplitude_is_divided_by_the_exner_function():
    """A 1 K temperature perturbation is more than 1 K of theta aloft.

    Two states differing only in pressure: at ``p = P0`` the Exner function
    is 1 and the theta increment equals the temperature amplitude; at
    ``p = P0/2`` it is larger by exactly ``2**RCP``.
    """
    at_p0 = _prepared_state(nx=48, ny=40, nz=12,
                            pressure=np.full((12, 40, 48), c.P0,
                                             dtype=np.float32))
    at_half = _prepared_state(nx=48, ny=40, nz=12,
                              pressure=np.full((12, 40, 48), 0.5 * c.P0,
                                               dtype=np.float32))
    cfg = _pcfg(_spec("t", 1.0), rh_cap=None)
    apply_perturbations(at_p0, 42, cfg)
    apply_perturbations(at_half, 42, cfg)

    draw, _ = gaussian_random_field(
        (12, 40, 48), seed=42, name="t", dx_km=1.0, dy_km=1.0,
        length_scale_km=4.0, vertical_scale_levels=0.0, xp=np)
    taper = boundary_taper(40, 48, 5, kind="cosine")
    expected = (draw * taper[None, :, :]).astype(np.float32)
    assert np.allclose(at_p0.thp, expected, rtol=0.0, atol=1e-6)

    interior = (slice(None), slice(6, 34), slice(6, 42))
    ratio = at_half.thp[interior] / at_p0.thp[interior]
    assert np.allclose(ratio, 2.0 ** c.RCP, rtol=1e-5)


def test_t_and_theta_together_are_a_configuration_error():
    with pytest.raises(ValueError, match="both perturb state.thp"):
        _pcfg(_spec("t", 1.0), _spec("theta", 1.0))


# --------------------------------------------------------------------------
# 8. fail-closed behaviour
# --------------------------------------------------------------------------

def test_temperature_perturbation_refuses_an_undiagnosed_pressure():
    state = DomainState(_cfg(), array_module=np)  # state.p is all zeros
    state.thb[...] = np.float32(300.0)
    with pytest.raises(ValueError, match="diagnosed positive pressure"):
        apply_perturbations(state, 1, _pcfg(_spec("t", 1.0), rh_cap=None))


def test_the_cap_refuses_an_undiagnosed_pressure():
    state = DomainState(_cfg(), array_module=np)
    state.thb[...] = np.float32(300.0)
    with pytest.raises(ValueError, match="diagnosed positive pressure"):
        apply_perturbations(state, 1, _pcfg(_spec("qv", 1.0e-4),
                                            rh_cap=1.0))


def test_perturbing_moisture_on_a_dry_state_is_refused():
    state = DomainState(_cfg(moist=False), array_module=np)
    state.p[...] = np.float32(5.0e4)
    state.thb[...] = np.float32(300.0)
    with pytest.raises(ValueError, match="state.qv is None"):
        apply_perturbations(state, 1, _pcfg(_spec("qv", 1.0e-4),
                                            rh_cap=None))


def test_a_length_scale_below_two_grid_cells_is_refused():
    state = _prepared_state()
    with pytest.raises(ValueError, match="below 2 grid spacings"):
        apply_perturbations(state, 1, _pcfg(_spec("theta", 1.0,
                                                  length_scale_km=1.5)))


def test_a_length_scale_that_spans_the_domain_is_refused():
    state = _prepared_state(nx=48, ny=40)
    with pytest.raises(ValueError, match="shorter domain span"):
        apply_perturbations(state, 1, _pcfg(_spec("theta", 1.0,
                                                  length_scale_km=30.0)))


def test_a_vertical_scale_that_spans_the_column_uses_padded_noise():
    state = _prepared_state(nz=12)
    provenance = apply_perturbations(state, 1, _pcfg(
        _spec("theta", 1.0, vertical_scale_levels=8.0)))
    column = provenance["fields"][0]["vertical_wrap"]
    assert column["fft_levels"] == 76
    assert column["physical_crop_levels"] == [32, 44]
    assert column["top_to_bottom_seam"] == pytest.approx(
        math.exp(-11.0 ** 2 / (2.0 * 8.0 ** 2)), abs=1e-12)


@pytest.mark.parametrize("kwargs,match", [
    (dict(name="T", amplitude=1.0, length_scale_km=4.0), "unknown"),
    (dict(name="t", amplitude=-1.0, length_scale_km=4.0), "non-negative"),
    (dict(name="t", amplitude=1.0, length_scale_km=0.0), "must be positive"),
    (dict(name="t", amplitude=float("nan"), length_scale_km=4.0),
     "must be finite"),
    (dict(name="t", amplitude=1.0, length_scale_km=4.0,
          vertical_scale_levels=-1.0), "non-negative"),
])
def test_field_spec_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FieldPerturbation(**kwargs)


@pytest.mark.parametrize("kwargs,match", [
    (dict(fields=()), "perturbs nothing"),
    (dict(rim_width=0), "rim_width must be an integer"),
    (dict(rim_taper="tophat"), "rim_taper must be"),
    (dict(qv_floor=-1.0), "qv_floor must be finite"),
    (dict(rh_cap=0.0), "rh_cap must be positive"),
    (dict(compute_dtype="float16"), "compute_dtype must be"),
    (dict(dx_km=0.0), "dx_km must be a positive"),
])
def test_config_validation(kwargs, match):
    base = dict(dx_km=1.0, dy_km=1.0, fields=(_spec("theta", 1.0),))
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        PerturbationConfig(**base)


def test_duplicate_fields_are_refused():
    with pytest.raises(ValueError, match="duplicate perturbation fields"):
        _pcfg(_spec("u", 1.0), _spec("u", 2.0))


def test_apply_perturbations_rejects_a_bare_mapping_for_cfg():
    state = _prepared_state()
    with pytest.raises(TypeError, match="must be a PerturbationConfig"):
        apply_perturbations(state, 1, {"dx_km": 1.0})


# --------------------------------------------------------------------------
# 9. configuration round-trip
# --------------------------------------------------------------------------

def test_config_from_a_table():
    cfg = PerturbationConfig.from_mapping({
        "dx_km": 3.0, "dy_km": 3.0, "rim_width": 4, "rim_taper": "linear",
        "rh_cap": 1.0,
        "fields": [
            {"name": "t", "amplitude": 0.5, "length_scale_km": 24.0},
            {"name": "qv", "amplitude": 0.0005, "length_scale_km": 24.0,
             "vertical_scale_levels": 4.0},
        ],
    })
    assert cfg.rim_width == 4
    assert cfg.rim_taper == "linear"
    assert cfg.field_names == ("t", "qv")
    assert cfg.spec("qv").vertical_scale_levels == 4.0


def test_config_from_a_table_keyed_by_field_name():
    cfg = PerturbationConfig.from_mapping({
        "dx_km": 1.0, "dy_km": 1.0,
        "fields": {"u": {"amplitude": 1.0, "length_scale_km": 8.0},
                   "v": {"amplitude": 1.0, "length_scale_km": 8.0}},
    })
    assert cfg.field_names == ("u", "v")


def test_unknown_config_keys_are_refused():
    with pytest.raises(ValueError, match="unknown perturbation config keys"):
        PerturbationConfig.from_mapping(
            {"dx_km": 1.0, "dy_km": 1.0, "fields": [], "amplitude": 2.0})


def test_fields_are_applied_in_a_canonical_order():
    """Listing order must not change the answer."""
    forward = _prepared_state(qv=1.0e-3)
    reverse = _prepared_state(qv=1.0e-3)
    specs = (_spec("theta", 1.0), _spec("qv", 1.0e-4), _spec("u", 1.0),
             _spec("v", 1.0))
    apply_perturbations(forward, 6, _pcfg(*specs))
    apply_perturbations(reverse, 6, _pcfg(*reversed(specs)))
    for name in ("thp", "qv", "u", "v"):
        assert np.array_equal(getattr(forward, name),
                              getattr(reverse, name)), name


# --------------------------------------------------------------------------
# 10. provenance
# --------------------------------------------------------------------------

def test_provenance_is_complete_and_serializable():
    state = _prepared_state(qv=1.0e-3)
    provenance = apply_perturbations(state, 20260730, _pcfg(
        _spec("theta", 1.0, vertical_scale_levels=3.0),
        _spec("qv", 5.0e-4), _spec("u", 1.5), _spec("v", 1.5),
        rim_width=4))

    assert provenance["schema"] == PROVENANCE_SCHEMA
    assert provenance["status"] == STATUS == "experimental"
    assert provenance["seed"] == 20260730
    assert provenance["backend"] == "numpy"
    assert provenance["fft_backend"] == "numpy"
    assert provenance["application_order"] == ["theta", "qv", "u", "v"]
    assert provenance["taper"]["rim_width_cells"] == 4
    assert provenance["taper"]["boundary_value"] == 0.0
    assert provenance["taper"]["interior_value"] == 1.0
    assert len(provenance["noise_sha256"]) == 64
    assert provenance["balance_not_imposed"]
    assert any("update_diagnostics" in line
               for line in provenance["post_conditions"])

    for record in provenance["fields"]:
        assert record["name"] in SUPPORTED_FIELDS
        assert len(record["noise_sha256"]) == 64
        assert len(record["stream_key_hex"]) == 32
        assert record["increment_rms"] > 0.0
        assert record["increment_min"] < 0.0 < record["increment_max"]
        assert record["units"]

    json.dumps(provenance)  # must round-trip into a run manifest


def test_provenance_records_the_amplitudes_it_actually_used():
    state = _prepared_state()
    provenance = apply_perturbations(
        state, 1, _pcfg(_spec("theta", 2.5, length_scale_km=6.0)))
    record = provenance["fields"][0]
    assert record["amplitude"] == 2.5
    assert record["length_scale_km"] == 6.0
    # Away from the rim the increment RMS tracks the requested 1-sigma; the
    # taper pulls the whole-field figure below it, which is the point.
    assert 0.5 * 2.5 < record["increment_rms"] < 2.5


def test_the_noise_hash_changes_with_the_seed_and_not_with_the_run():
    first = apply_perturbations(_prepared_state(), 1,
                                _pcfg(_spec("theta", 1.0)))
    again = apply_perturbations(_prepared_state(), 1,
                                _pcfg(_spec("theta", 1.0)))
    other = apply_perturbations(_prepared_state(), 2,
                                _pcfg(_spec("theta", 1.0)))
    assert first["noise_sha256"] == again["noise_sha256"]
    assert first["noise_sha256"] != other["noise_sha256"]


# --------------------------------------------------------------------------
# 11. documented stubs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stub", [recycled_difference_perturbations])
def test_unbuilt_routes_raise_rather_than_falling_back(stub):
    with pytest.raises(NotImplementedError, match="non-goal"):
        stub()


def test_member_boundaries_are_built_not_a_stub():
    """Audit S4: perturbed_lateral_boundaries was a stub; it is built now
    (tests/test_da_perturb_boundaries.py holds it to its claims)."""
    with pytest.raises(TypeError):
        perturbed_lateral_boundaries()


def test_importing_the_module_does_not_import_cupy():
    """The NumPy fallback has to be reachable on a host with no CUDA at all.

    Run in a subprocess: by the time this file's other tests have run, cupy
    may already be in ``sys.modules`` for unrelated reasons, and asking the
    current interpreter would prove nothing.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import gpuwm.da.perturb; "
         "print('cupy' in sys.modules)"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "gpuwm.da.perturb pulled in cupy at import time; a CPU-only host "
        f"can no longer use it. stdout={result.stdout!r}")


def test_the_module_documents_what_it_does_not_balance():
    from gpuwm.da import perturb
    text = perturb.__doc__
    # The boundary line changed with audit S4: the forcing is perturbed by
    # a separate call (perturbed_lateral_boundaries), and the docstring
    # says so rather than calling it a non-goal.
    # 2026-10-06: the wind is non-divergent and the mass hydrostatic by
    # default; what remains un-imposed is the coupling between them.
    for phrase in ("Wind: non-divergent by construction",
                   "Mass: hydrostatic",
                   "No geostrophic or gradient-wind coupling",
                   "The map factor is not seen",
                   "The boundary forcing is perturbed by a separate call"):
        assert phrase in text


# --------------------------------------------------------------------------
# 12. CuPy
# --------------------------------------------------------------------------

@requires_gpu
def test_cupy_backend_matches_the_host_length_scale_and_noise_stamp():
    """The device path is the same perturbation, not a similar one.

    The white noise is drawn on the host in both cases, so ``noise_sha256``
    must match exactly whichever FFT ran.  The fields themselves are only
    compared to single precision, because cuFFT and pocketfft do not round
    identically and this module has never claimed they do.
    """
    import cupy as cp

    shape = (8, 128, 128)
    host, host_info = gaussian_random_field(
        shape, seed=101, name="t", dx_km=1.0, dy_km=1.0,
        length_scale_km=4.0, vertical_scale_levels=0.0, xp=np)
    device, device_info = gaussian_random_field(
        shape, seed=101, name="t", dx_km=1.0, dy_km=1.0,
        length_scale_km=4.0, vertical_scale_levels=0.0, xp=cp)

    assert device_info["noise_sha256"] == host_info["noise_sha256"]
    assert device_info["backend"] == "cupy"
    assert host_info["fft_backend"] == "numpy"
    assert device_info["fft_backend"] in ("cupy", "numpy")
    assert isinstance(device, cp.ndarray), (
        "a cupy request must return a device array even when the FFT fell "
        "back to the host")
    assert np.allclose(cp.asnumpy(device), host, rtol=1e-6, atol=1e-7)

    k, energy = radial_power_spectrum(device, 1.0, bins=32)
    assert fit_gaussian_length_scale(k, energy) == pytest.approx(4.0,
                                                                 rel=0.05)


@requires_gpu
def test_the_fft_backend_is_reported_not_assumed():
    """If cuFFT will not load, the provenance says ``numpy`` next to a
    ``cupy`` backend rather than claiming a device FFT that never ran."""
    import cupy as cp

    from gpuwm.da.perturb import _device_fft_available

    _, info = gaussian_random_field(
        (4, 32, 32), seed=1, name="t", dx_km=1.0, dy_km=1.0,
        length_scale_km=4.0, xp=cp)
    expected = "cupy" if _device_fft_available(cp) else "numpy"
    assert info["fft_backend"] == expected


@requires_gpu
def test_apply_perturbations_on_a_device_state():
    import cupy as cp

    cfg = _cfg(nx=48, ny=40, nz=12)
    state = DomainState(cfg, array_module=cp)
    column = cp.asarray(np.linspace(1.0e5, 2.0e4, 12, dtype=np.float32))
    state.p[...] = column[:, None, None]
    state.thb[...] = cp.float32(300.0)
    state.qv[...] = cp.float32(1.0e-3)

    provenance = apply_perturbations(state, 55, _pcfg(
        _spec("theta", 1.0), _spec("qv", 5.0e-4), _spec("u", 1.5),
        _spec("v", 1.5)))
    assert provenance["backend"] == "cupy"
    assert provenance["fft_backend"] in ("cupy", "numpy")
    assert float(state.qv.min()) >= 0.0
    assert state.u.shape == (12, 40, 49)
    assert state.v.shape == (12, 41, 48)
    for name in ("thp", "qv", "u", "v"):
        array = getattr(state, name)
        assert array.dtype == cp.float32
        assert bool(cp.all(cp.isfinite(array)))
        assert int(cp.count_nonzero(array)) > 0
    # Boundary rows untouched: theta' started at exactly zero there.
    for edge in (state.thp[:, 0, :], state.thp[:, -1, :],
                 state.thp[:, :, 0], state.thp[:, :, -1]):
        assert float(cp.abs(edge).max()) == 0.0
    for edge in (state.u[:, :, 0], state.u[:, :, -1],
                 state.v[:, 0, :], state.v[:, -1, :]):
        assert float(cp.abs(edge).max()) == 0.0


@requires_gpu
def test_mixed_backend_state_is_refused():
    import cupy as cp

    cfg = _cfg()
    state = DomainState(cfg, array_module=cp)
    state.p[...] = cp.float32(5.0e4)
    state.thb[...] = cp.float32(300.0)
    state.u = np.zeros((12, 40, 49), dtype=np.float32)  # host array on a
    with pytest.raises(TypeError, match="different array backend"):
        apply_perturbations(state, 1, _pcfg(_spec("u", 1.0), rh_cap=None,
                                            wind_mode="independent"))


# --------------------------------------------------------------------------
# 13. balance (2026-10-06): non-divergent wind, hydrostatic mass
# --------------------------------------------------------------------------

def _wind_cfg(amplitude=2.0, length_km=6.0, **kwargs):
    kwargs.setdefault("dx_km", 1.0)
    kwargs.setdefault("dy_km", 1.0)
    return _pcfg(_spec("u", amplitude, length_km),
                 _spec("v", amplitude, length_km), **kwargs)


def _c_grid_divergence(u, v, dx_m, dy_m):
    """``(u[i+1] - u[i]) / dx + (v[j+1] - v[j]) / dy`` at every mass point."""
    u64, v64 = u.astype(np.float64), v.astype(np.float64)
    return ((u64[:, :, 1:] - u64[:, :, :-1]) / dx_m
            + (v64[:, 1:, :] - v64[:, :-1, :]) / dy_m)


def test_rotational_wind_increment_has_zero_c_grid_divergence():
    """The defect this exists for: two unrelated u/v draws put half their
    kinetic energy into divergence, which the model radiates as gravity
    waves in the first hour (the DA lanes measured the members' spread
    3-5x below O-B and 3.4x the control's surface pressure tendency).
    One streamfunction, differenced the C-grid way, is non-divergent on
    the model's own stencil wherever the rim taper is one.  Inside the
    rim band the component taper leaves ``u * grad(taper)`` and no more
    (the taper is on the components, not on psi: see
    test_rotational_rim_band_carries_no_more_than_the_spec)."""
    rim = 5
    state = _prepared_state(nx=64, ny=56, nz=6)
    apply_perturbations(state, 11, _wind_cfg(compute_dtype="float64",
                                             rim_width=rim))
    div = _c_grid_divergence(state.u, state.v, 1000.0, 1000.0)
    scale = np.abs(state.u).max() / 1000.0          # one-cell shear
    assert scale > 1.0e-6, "the wind was not perturbed"
    # Mass points whose four faces all sit where the taper is one.
    interior = div[:, rim:-rim, rim:-rim]
    # float32 state words: rounding at ~1e-7 of the shear, nothing more
    assert np.abs(interior).max() <= 1.0e-5 * scale, (
        f"max interior |div| {np.abs(interior).max():.3e} vs shear "
        f"{scale:.3e}")
    # The band: the cosine taper's steepest slope is pi/(2 rim dx), and
    # the wind under it is at most a few sigma.
    band_limit = 5.0 * 2.0 / (rim * 1000.0)          # amplitude 2.0
    assert np.abs(div).max() <= band_limit, (
        f"rim-band |div| {np.abs(div).max():.3e} exceeds u*grad(taper) "
        f"{band_limit:.3e}")


def test_rotational_rim_band_carries_no_more_than_the_spec():
    """The defect the component taper prevents (2026-10-06, found in
    review): with the taper on the streamfunction, a non-divergent field
    that vanishes at the rim has psi constant there, so psi (amplitude
    sigma_u * L_psi) drops to that constant across the rim band and the
    tangential wind in the band is sigma_u * L_psi / W.  With the DA
    cycle tool's numbers (1.5 m/s at 150 km, a 5-cell rim at 3 km) that
    measured 35x the configured amplitude (unit row RMS 35, peak 47):
    50-70 m/s of shear along every boundary of every member.  Here, the
    cycle tool's own scale, spacing and rim on a span that resolves it:
    no row or column of either component may carry more than the
    interior does, within sampling."""
    rim, dx_km, length_km, amplitude = 5, 3.0, 150.0, 1.5
    u_rows, v_cols, u_in, v_in = [], [], [], []
    for seed in (1, 2):
        state = _prepared_state(nx=400, ny=400, nz=4)
        apply_perturbations(state, seed, _wind_cfg(
            amplitude, length_km, dx_km=dx_km, dy_km=dx_km, rim_width=rim,
            compute_dtype="float64"))
        u = state.u.astype(np.float64)
        v = state.v.astype(np.float64)
        u_rows.append(np.mean(u ** 2, axis=(0, 2)))      # per row j
        v_cols.append(np.mean(v ** 2, axis=(0, 1)))      # per column i
        u_in.append(np.mean(u[:, rim + 2:-rim - 2, rim + 2:-rim - 2] ** 2))
        v_in.append(np.mean(v[:, rim + 2:-rim - 2, rim + 2:-rim - 2] ** 2))
    u_row_rms = np.sqrt(np.mean(u_rows, axis=0))
    v_col_rms = np.sqrt(np.mean(v_cols, axis=0))
    u_interior = math.sqrt(float(np.mean(u_in)))
    v_interior = math.sqrt(float(np.mean(v_in)))
    assert 0.7 * amplitude <= u_interior <= 1.3 * amplitude, u_interior
    # The band (the tapered rows and columns plus one): with the taper on
    # psi it carried 35x the interior; with it on the components it
    # carries at most the interior, within the sampling of a 150 km field
    # on a 1200 km span.
    band = rim + 2
    u_band = np.concatenate([u_row_rms[:band], u_row_rms[-band:]])
    v_band = np.concatenate([v_col_rms[:band], v_col_rms[-band:]])
    assert u_band.max() <= 1.5 * u_interior, (
        f"u rim-band row RMS peaks at {u_band.max():.2f} against an "
        f"interior of {u_interior:.2f}: {np.round(u_band, 2)}")
    assert v_band.max() <= 1.5 * v_interior, (
        f"v rim-band column RMS peaks at {v_band.max():.2f} against an "
        f"interior of {v_interior:.2f}: {np.round(v_band, 2)}")
    # and the rim rows themselves are zero (the invariant)
    assert u_row_rms[0] == 0.0 and u_row_rms[-1] == 0.0
    assert v_col_rms[0] == 0.0 and v_col_rms[-1] == 0.0


def test_independent_draws_are_half_divergent_and_rotational_are_not():
    """Names the breakage in numbers: the divergent fraction of the
    kinetic energy, from the RMS divergence against the RMS vorticity
    on the same stencil (equal for two unrelated isotropic draws)."""
    rim = 5

    def fractions(cfg):
        state = _prepared_state(nx=96, ny=96, nz=4)
        apply_perturbations(state, 23, cfg)
        u, v = state.u.astype(np.float64), state.v.astype(np.float64)
        div = _c_grid_divergence(u, v, 1000.0, 1000.0)
        # vorticity on the interior corners the same way
        dv_dx = (v[:, :, 1:] - v[:, :, :-1])[:, 1:-1, :] / 1000.0
        du_dy = (u[:, 1:, :] - u[:, :-1, :])[:, :, 1:-1] / 1000.0
        vort = dv_dx - du_dy
        return (np.sqrt(np.mean(div ** 2)), np.sqrt(np.mean(vort ** 2)),
                np.abs(div[:, rim:-rim, rim:-rim]).max())

    div_i, vort_i, _ = fractions(_wind_cfg(wind_mode="independent",
                                           compute_dtype="float64",
                                           rim_width=rim))
    div_r, vort_r, interior_r = fractions(_wind_cfg(compute_dtype="float64",
                                                    rim_width=rim))
    assert 0.7 <= div_i / vort_i <= 1.3, (div_i, vort_i)   # half and half
    # Rotational: nothing where the taper is one, and over the whole
    # field (rim band included, where the component taper puts
    # u * grad(taper) into the divergence) well under half.
    assert interior_r <= 1.0e-5 * vort_r, (interior_r, vort_r)
    assert div_r / vort_r <= 0.5 * (div_i / vort_i), (div_r, vort_r,
                                                      div_i, vort_i)


def test_independent_wind_mode_is_the_previous_draw_bit_for_bit():
    """The comparison arm reproduces the pre-2026-10-06 members: same
    stream (keyed on the field name), same bytes."""
    state = _prepared_state(nx=48, ny=40, nz=6)
    provenance = apply_perturbations(
        state, 5, _wind_cfg(wind_mode="independent"))
    for name in ("u", "v"):
        record = next(r for r in provenance["fields"] if r["name"] == name)
        shape = tuple(record["shape"])
        _, info = gaussian_random_field(shape, seed=5, name=name, dx_km=1.0,
                                        dy_km=1.0, length_scale_km=6.0, xp=np)
        assert record["noise_sha256"] == info["noise_sha256"]
        assert record["wind_mode"] == "independent"
        assert "streamfunction" not in record
    assert provenance["wind_mode"] == "independent"


def test_rotational_wind_amplitude_matches_the_spec():
    """One scaling for both components: u carries the configured 1-sigma
    exactly in expectation, v within its analytic ratio (one, on a square
    grid of equal extents).  Measured over 24 members on the untapered
    interior; the band is sampling, not a fudge."""
    amplitude, seeds = 2.0, range(24)
    u_var, v_var, ratio = [], [], None
    for seed in seeds:
        state = _prepared_state(nx=96, ny=96, nz=1)
        prov = apply_perturbations(state, seed, _wind_cfg(amplitude, 6.0))
        rim = 5 + 2
        u_var.append(np.mean(state.u[:, rim:-rim, rim:-rim]
                             .astype(np.float64) ** 2))
        v_var.append(np.mean(state.v[:, rim:-rim, rim:-rim]
                             .astype(np.float64) ** 2))
        ratio = next(r for r in prov["fields"] if r["name"] == "v")[
            "streamfunction"]["v_amplitude_ratio"]
    u_sigma, v_sigma = np.sqrt(np.mean(u_var)), np.sqrt(np.mean(v_var))
    assert abs(ratio - 1.0) < 1.0e-6                      # square, equal
    assert 0.9 <= u_sigma / amplitude <= 1.1, u_sigma
    assert 0.9 <= v_sigma / amplitude <= 1.1, v_sigma


def test_rotational_wind_spectrum_peaks_at_the_prescribed_scale():
    """The module's length-scale contract, kept for the wind: the
    streamfunction is drawn at sqrt(3) L so the WIND's radial spectrum
    peaks at k = 1/L (a derivative of a Gaussian field peaks at
    sqrt(3)/L_psi).  Averaged over four seeds; the band is the one the
    scalar test uses."""
    from gpuwm.da.perturb import rotational_wind_draw
    energy = None
    for seed in (1, 2, 3, 17):
        u, _v, _info = rotational_wind_draw(
            SPECTRAL_SHAPE, seed=seed, dx_km=SPECTRAL_DX_KM,
            dy_km=SPECTRAL_DX_KM, length_scale_km=SPECTRAL_L_KM, xp=np)
        k, e = radial_power_spectrum(u[:, :, :-1], SPECTRAL_DX_KM,
                                     bins=SPECTRAL_BINS)
        energy = e if energy is None else energy + e
    peak = spectral_peak_wavenumber(k, energy)
    ratio = peak * SPECTRAL_L_KM
    assert 0.85 <= ratio <= 1.15, f"wind spectral peak ratio {ratio:.4f}"


def test_rotational_wind_rim_rows_are_untouched_and_the_interior_moves():
    state = _prepared_state(nx=48, ny=40, nz=6)
    state.u[...] = np.float32(-0.0)
    state.v[...] = np.float32(-0.0)
    before = {"u": state.u.copy(), "v": state.v.copy()}
    apply_perturbations(state, 3, _wind_cfg())
    for name in ("u", "v"):
        now, was = getattr(state, name), before[name]
        for sl in (np.s_[:, 0, :], np.s_[:, -1, :], np.s_[:, :, 0],
                   np.s_[:, :, -1]):
            assert now[sl].tobytes() == was[sl].tobytes(), name
        assert np.count_nonzero(now) > 0.5 * now.size, name


def test_rotational_mode_refuses_a_lone_component_and_mismatched_specs():
    with pytest.raises(ValueError, match="derives u AND v"):
        _pcfg(_spec("u", 1.0))
    with pytest.raises(ValueError, match="must agree on amplitude"):
        _pcfg(_spec("u", 1.0), _spec("v", 2.0))
    with pytest.raises(ValueError, match="must agree on length_scale_km"):
        _pcfg(_spec("u", 1.0, 4.0), _spec("v", 1.0, 8.0))
    with pytest.raises(ValueError, match="unknown wind_mode"):
        _pcfg(_spec("u", 1.0), _spec("v", 1.0), wind_mode="sideways")
    # the comparison arm takes a lone component
    _pcfg(_spec("u", 1.0), wind_mode="independent")


def test_rotational_provenance_records_the_streamfunction():
    state = _prepared_state(nx=48, ny=40, nz=6)
    prov = apply_perturbations(state, 8, _wind_cfg(2.0, 6.0))
    json.dumps(prov)
    u = next(r for r in prov["fields"] if r["name"] == "u")
    v = next(r for r in prov["fields"] if r["name"] == "v")
    assert u["wind_mode"] == v["wind_mode"] == "rotational"
    assert u["streamfunction"]["noise_sha256"] == v["streamfunction"][
        "noise_sha256"]
    assert u["noise_sha256"] == v["noise_sha256"]            # one draw
    assert u["streamfunction"]["shape"] == [6, 41, 49]
    assert abs(u["streamfunction"]["length_scale_km"]
               - math.sqrt(3.0) * 6.0) < 1e-12
    assert "taper is one" in u["streamfunction"]["stencil"]
    assert any("non-divergent" in line or "divergence of the increment"
               in line for line in prov["balance_imposed"])


def _column_state(nx=40, ny=32, nz=12, *, dtype=np.float32):
    """A host state with a loaded sigma coordinate, enough for the
    hydrostatic column integration (hypsometric_opt = 1): the arrays
    :mod:`gpuwm.da.hydrostatic` reads, nothing the dycore would."""
    from types import SimpleNamespace
    eta = np.linspace(1.0, 0.0, nz + 1)
    dnw = np.diff(eta)                                   # negative, WRF's
    rdnw = 1.0 / dnw
    dn = np.zeros(nz + 1)
    dn[1:nz] = 0.5 * (dnw[:-1] + dnw[1:])
    rdn = np.zeros(nz + 1)
    rdn[1:nz] = 1.0 / dn[1:nz]
    half = 0.5 * (eta[:-1] + eta[1:])
    p_top, mub = 5000.0, 9.5e4
    pressure = (p_top + half * mub).astype(np.float64)
    z = lambda *s: np.zeros(s, dtype=dtype)  # noqa: E731
    state = SimpleNamespace(
        thp=z(nz, ny, nx), qv=np.full((nz, ny, nx), 2.0e-3, dtype),
        u=z(nz, ny, nx + 1), v=z(nz, ny + 1, nx), php=z(nz + 1, ny, nx),
        p=np.broadcast_to(pressure[:, None, None].astype(dtype),
                          (nz, ny, nx)).copy(),
        mup=z(ny, nx), mub2d=np.full((ny, nx), mub, dtype),
        thb=np.full(nz, 300.0, dtype),
        c1h=np.ones(nz, dtype), c2h=np.zeros(nz, dtype),
        c1f=np.ones(nz + 1, dtype), c2f=np.zeros(nz + 1, dtype),
        dnw=dnw.astype(dtype), rdnw=rdnw.astype(dtype),
        rdn=rdn.astype(dtype), p_top=p_top)
    return state


def _theta_cfg(**kwargs):
    kwargs.setdefault("dx_km", 1.0)
    kwargs.setdefault("dy_km", 1.0)
    kwargs.setdefault("rim_width", 4)
    kwargs.setdefault("hypsometric_opt", 1)
    return PerturbationConfig(fields=(_spec("theta", 1.0, 4.0),), **kwargs)


def test_hydrostatic_mass_balance_re_integrates_php_where_theta_moved():
    """The other half of the defect: a theta draw under a fixed
    geopotential reads as a pressure jump of about 4 hPa per kelvin
    (gpuwm.da.hydrostatic), which the first acoustic substeps answer
    with a ring.  By default the column's php is re-integrated at its
    own dry mass: zero at the surface, moving above exactly where theta
    moved, and byte-identical in every column it did not reach."""
    state = _column_state()
    php_before = state.php.copy()
    prov = apply_perturbations(state, 4, _theta_cfg())
    receipt = prov["mass_balance"]
    assert receipt["applied"] and receipt["mode"] == "hydrostatic"
    assert receipt["perturbed_column_fields"] == ["theta"]
    assert receipt["hypsometric_opt"] == 1
    assert receipt["max_abs_dphp_m2s2"] > 0.0
    assert np.array_equal(state.php[0], php_before[0])      # terrain
    moved = state.thp != 0.0
    columns = moved.any(axis=0)
    assert columns.sum() > 0.5 * columns.size
    dphp = state.php.astype(np.float64) - php_before
    assert np.count_nonzero(dphp[1:][:, columns]) > 0
    assert receipt["columns_moved"] == int(columns.sum())
    # untouched columns (the rim): the words did not move
    assert state.php[:, ~columns].tobytes() == php_before[:, ~columns].tobytes()
    # The integration itself, restated independently: a warm layer is a
    # thicker layer, d(phi_k) = -dnw_k * mu * d(alt_k) with alt from the
    # equation of state at the unchanged pressure (no loading change),
    # summed up from the terrain.
    thb = state.thb.astype(np.float64)[:, None, None]
    p = state.p.astype(np.float64)
    qv = state.qv.astype(np.float64)
    mu = (state.mub2d.astype(np.float64) + state.mup.astype(np.float64))

    def alt(thp):
        theta = thb + thp
        return (c.RD * theta * (1.0 + c.RVOVRD * qv)
                * (p / c.P0) ** c.RCP / p)

    dalt = alt(state.thp.astype(np.float64)) - alt(0.0)
    op = -state.dnw.astype(np.float64)[:, None, None] * mu[None]
    expected = np.cumsum(op * dalt, axis=0)
    np.testing.assert_allclose(dphp[1:], expected, rtol=2e-5, atol=1e-3)
    # and the size is the hydrostatic one: about g * H * dtheta / theta,
    # a few hundred m2/s2 per kelvin over a 10 km column at 300 K
    gain = (np.abs(dphp[-1][columns]).max()
            / np.abs(state.thp.astype(np.float64)).max())
    assert 50.0 < gain < 2000.0, gain


def test_mass_balance_none_leaves_php_alone():
    state = _column_state()
    php_before = state.php.copy()
    prov = apply_perturbations(state, 4, _theta_cfg(mass_balance="none"))
    assert prov["mass_balance"] == {
        "applied": False, "mode": "none",
        "reason": "mass_balance = 'none' (configured)"}
    assert np.array_equal(state.php, php_before)
    assert np.count_nonzero(state.thp) > 0


def test_mass_balance_refuses_a_state_without_a_vertical_coordinate():
    """``_prepared_state`` has dnw all zero: every layer operator would be
    zero and the balance a silent no-op.  Refused by name, before any
    field is written."""
    state = _prepared_state()
    thp = state.thp.copy()
    with pytest.raises(ValueError, match="dnw is absent or all zero"):
        apply_perturbations(state, 1, _pcfg(_spec("theta", 1.0),
                                            mass_balance="hydrostatic",
                                            hypsometric_opt=1))
    assert np.array_equal(state.thp, thp)


def test_mass_balance_needs_the_hypsometric_option():
    state = _column_state()
    with pytest.raises(ValueError, match="needs hypsometric_opt"):
        apply_perturbations(state, 1, _theta_cfg(hypsometric_opt=None))
    with pytest.raises(ValueError, match="hypsometric_opt must be 1 or 2"):
        _theta_cfg(hypsometric_opt=3)
    with pytest.raises(ValueError, match="needs state.p_top"):
        state.p_top = None
        apply_perturbations(state, 1, _theta_cfg(hypsometric_opt=2))


def test_mass_balance_is_skipped_with_a_reason_for_a_wind_only_draw():
    """A wind-only draw changes no column, so the default needs neither
    hypsometric_opt nor a loaded base, and says why it did nothing."""
    state = _prepared_state()
    prov = apply_perturbations(state, 1, _wind_cfg(mass_balance="hydrostatic"))
    assert prov["mass_balance"]["applied"] is False
    assert "no thermodynamic" in prov["mass_balance"]["reason"]


def test_config_table_accepts_the_balance_keys():
    cfg = PerturbationConfig.from_mapping({
        "dx_km": 3.0, "dy_km": 3.0,
        "fields": [{"name": "u", "amplitude": 1.5, "length_scale_km": 150},
                   {"name": "v", "amplitude": 1.5, "length_scale_km": 150}],
        "wind_mode": "independent", "mass_balance": "none",
        "hypsometric_opt": 2,
    })
    assert (cfg.wind_mode, cfg.mass_balance, cfg.hypsometric_opt) == (
        "independent", "none", 2)
    default = PerturbationConfig.from_mapping({
        "dx_km": 3.0, "dy_km": 3.0,
        "fields": [{"name": "theta", "amplitude": 0.5,
                    "length_scale_km": 60}]})
    assert (default.wind_mode, default.mass_balance,
            default.hypsometric_opt) == ("rotational", "hydrostatic", None)
    with pytest.raises(ValueError, match="unknown mass_balance"):
        PerturbationConfig.from_mapping({
            "dx_km": 3.0, "dy_km": 3.0, "mass_balance": "maybe",
            "fields": [{"name": "theta", "amplitude": 0.5,
                        "length_scale_km": 60}]})
