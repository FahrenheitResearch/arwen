"""Per-member lateral boundaries and additive inflation (gpuwm.da.perturb).

Audit S4: spread came from one draw at the start plus RTPS, and members
shared one boundary, so spread died toward the rim.  These tests hold the
two remedies to what their docstrings claim.
"""

import math

import numpy as np
import pytest

from gpuwm.da.perturb import (
    PerturbationConfig,
    additive_inflation,
    boundary_taper,
    gaussian_random_field,
    perturbed_lateral_boundaries,
)
from gpuwm.ingest.lateral_bc import (
    build_lateral_boundaries,
    evaluate_boundary_side,
)

NZ, NY, NX = 8, 40, 48
WIDTH = 5
HOUR = 3600.0


def _cfg(qv_mode="lognormal"):
    fields = [
        {"name": "u", "amplitude": 1.5, "length_scale_km": 6.0},
        {"name": "v", "amplitude": 1.5, "length_scale_km": 6.0},
        {"name": "theta", "amplitude": 0.5, "length_scale_km": 6.0,
         "vertical_scale_levels": 1.0},
        {"name": "qv", "amplitude": 0.05 if qv_mode == "lognormal" else 1e-4,
         "length_scale_km": 6.0, "mode": qv_mode},
    ]
    return PerturbationConfig.from_mapping({
        "dx_km": 1.0, "dy_km": 1.0, "rim_width": WIDTH, "fields": fields})


def _snapshot(offset):
    rng = np.random.default_rng(int(offset * 10))
    return {
        "u": 1.0e5 * (5.0 + offset + rng.normal(0, 0.1, (NZ, NY, NX + 1))),
        "v": 1.0e5 * (2.0 + rng.normal(0, 0.1, (NZ, NY + 1, NX))),
        "theta": 1.0e5 * (10.0 + offset + rng.normal(0, 0.1, (NZ, NY, NX))),
        "phi": 1.0e5 * rng.normal(0, 1.0, (NZ + 1, NY, NX)),
        "mu": rng.normal(0, 10.0, (NY, NX)),
        "qv": 1.0e5 * np.full((NZ, NY, NX), 8.0e-3 + 1e-3 * offset),
    }


def _boundaries(frames=3):
    return build_lateral_boundaries(
        [_snapshot(i) for i in range(frames)],
        [float(i) * HOUR for i in range(frames)], spec_bdy_width=WIDTH)


def _coupling():
    return {"u": np.full((NZ, NY, NX + 1), 1.0e5),
            "v": np.full((NZ, NY + 1, NX), 1.0e5),
            "theta": np.full((NZ, NY, NX), 1.0e5),
            "qv": np.full((NZ, NY, NX), 1.0e5)}


def _member(seed=7, scale=1.0, cfg=None, base=None):
    return perturbed_lateral_boundaries(
        base or _boundaries(), cfg or _cfg(), seed=seed,
        coupling=_coupling(), scale=scale)


def test_scale_zero_is_the_shared_boundary_itself():
    base = _boundaries()
    same, record = perturbed_lateral_boundaries(
        base, _cfg(), seed=1, coupling=_coupling(), scale=0.0)
    assert same is base
    assert "unchanged" in record["note"]


def test_the_start_frame_agrees_with_the_shared_boundary():
    """Frame 0 is the run's start, where the member's own initial state is
    tapered to the shared boundary at the rim; the forcing must agree."""
    base = _boundaries()
    member, _ = _member(base=base)
    for table in ("u", "v", "theta", "qv"):
        for side in ("west", "east", "south", "north"):
            np.testing.assert_array_equal(
                getattr(member.intervals[0].fields[table], side).value,
                getattr(base.intervals[0].fields[table], side).value)


def test_the_perturbation_is_continuous_across_intervals_and_nonzero():
    base = _boundaries()
    member, _ = _member(base=base)
    for table in ("u", "theta", "qv"):
        for side in ("west", "north"):
            end0 = evaluate_boundary_side(
                getattr(member.intervals[0].fields[table], side), HOUR)[0]
            start1 = getattr(member.intervals[1].fields[table], side).value
            np.testing.assert_allclose(end0, start1, rtol=1e-12)
            base_start1 = getattr(base.intervals[1].fields[table],
                                  side).value
            assert np.abs(start1 - base_start1).max() > 0.0


def test_frame_one_is_the_members_own_initial_draw():
    """The forcing ramps into the pattern apply_perturbations drew."""
    cfg = _cfg()
    base = _boundaries()
    member, _ = _member(seed=11, base=base, cfg=cfg)
    draw, _ = gaussian_random_field(
        (NZ, NY, NX + 1), seed=11, name="u", dx_km=1.0, dy_km=1.0,
        length_scale_km=6.0, xp=np, fft_host=True)
    added = (member.intervals[1].fields["u"].west.value
             - base.intervals[1].fields["u"].west.value)
    np.testing.assert_allclose(added, 1.0e5 * 1.5 * draw[..., :WIDTH],
                               rtol=1e-9, atol=1e-6)


def test_members_differ_and_a_member_is_reproducible():
    one, record_one = _member(seed=3)
    again, record_again = _member(seed=3)
    other, record_other = _member(seed=4)
    assert record_one["tables_sha256"] == record_again["tables_sha256"]
    assert record_one["tables_sha256"] != record_other["tables_sha256"]


def test_untouched_tables_and_positive_vapour():
    base = _boundaries()
    member, record = _member(base=base)
    assert set(record["not_perturbed"]) == {"mu", "phi"}
    for interval_m, interval_b in zip(member.intervals, base.intervals):
        assert interval_m.fields["phi"] is interval_b.fields["phi"]
        assert interval_m.fields["mu"] is interval_b.fields["mu"]
        for side in ("west", "east", "south", "north"):
            value = getattr(interval_m.fields["qv"], side)
            for t in (0.0, 0.5 * HOUR, HOUR):
                assert evaluate_boundary_side(value, t)[0].min() > 0.0


def test_the_time_correlation_keeps_later_frames_unit_variance():
    member, record = _member(base=_boundaries(frames=5))
    u = next(entry for entry in record["fields"] if entry["name"] == "u")
    rms = u["uncoupled_rms_per_frame"]
    assert len(rms) == 4
    for value in rms:
        assert 0.5 * 1.5 < value < 1.6 * 1.5


# ---------------------------------------------------------------- additive

def _priors(members=6):
    rng = np.random.default_rng(5)
    return {m: {"u": rng.normal(0, 1, (NZ, NY, NX + 1)).astype(np.float32),
                "v": rng.normal(0, 1, (NZ, NY + 1, NX)).astype(np.float32),
                "thp": rng.normal(0, 1, (NZ, NY, NX)).astype(np.float32),
                "qv": np.full((NZ, NY, NX), 8e-3, np.float32)}
            for m in range(members)}


def test_additive_inflation_keeps_the_mean_and_spares_the_rim():
    noise, record = additive_inflation(_priors(), _cfg(), seed=9, leg=2,
                                       scale=0.25)
    assert set(noise) == set(range(6))
    taper = boundary_taper(NY, NX, WIDTH, xp=np)
    for attribute in ("u", "v", "thp", "qv"):
        stack = np.stack([noise[m][attribute] for m in noise]).astype(
            np.float64)
        scale = np.abs(stack).max()
        assert scale > 0.0
        assert np.abs(stack.mean(axis=0)).max() < 1e-6 * scale
        assert stack.dtype == np.float64
        assert noise[0][attribute].dtype == np.float32
    thp = np.stack([noise[m]["thp"] for m in noise])
    assert np.all(thp[:, :, taper == 0.0] == 0.0)
    fields = {entry["name"]: entry for entry in record["fields"]}
    assert fields["u"]["amplitude"] == pytest.approx(0.25 * 1.5)
    assert 0.3 * 0.375 < fields["u"]["added_rms"] < 1.5 * 0.375
    assert record["mean_removed"] is True


def test_additive_inflation_is_reproducible_and_leg_dependent():
    first, _ = additive_inflation(_priors(), _cfg(), seed=9, leg=2,
                                  scale=0.25)
    again, _ = additive_inflation(_priors(), _cfg(), seed=9, leg=2,
                                  scale=0.25)
    later, _ = additive_inflation(_priors(), _cfg(), seed=9, leg=3,
                                  scale=0.25)
    assert np.array_equal(first[1]["u"], again[1]["u"])
    assert not np.array_equal(first[1]["u"], later[1]["u"])


def test_additive_inflation_off_and_single_member_add_nothing():
    assert additive_inflation(_priors(), _cfg(), seed=1, leg=0,
                              scale=0.0)[0] == {}
    assert additive_inflation(_priors(1), _cfg(), seed=1, leg=0,
                              scale=0.25)[0] == {}
    with pytest.raises(ValueError):
        additive_inflation(_priors(), _cfg(), seed=1, leg=0, scale=-0.1)


def test_a_state_without_eager_tables_is_named_not_refused():
    """The cycle driver records why and runs the shared tables."""
    from types import SimpleNamespace

    from gpuwm.da.perturb import boundary_perturbation_unavailable

    assert boundary_perturbation_unavailable(_boundaries()) is None
    assert "no lateral boundary" in boundary_perturbation_unavailable(None)
    streamed = SimpleNamespace(intervals=SimpleNamespace(bounds=((0, 1),)))
    assert "streamed" in boundary_perturbation_unavailable(streamed)
    assert "holds no interval" in boundary_perturbation_unavailable(
        SimpleNamespace())


def test_echo_weight_marks_observed_echo_and_fades_around_it():
    """Audit S5: the echo-located noise sits where the radar sees echo."""
    from gpuwm.da.perturb import echo_weight

    z = np.full((NZ, NY, NX), -10.0)
    mask = np.ones((NZ, NY, NX), bool)
    z[2, 18:22, 20:24] = 40.0
    w = echo_weight(z, mask, dx_km=1.0, dy_km=1.0, length_scale_km=3.0)
    assert w.shape == (NY, NX)
    assert np.all(w[18:22, 20:24] == 1.0)
    assert w[0, 0] == 0.0 and 0.0 < w[16, 21] < 1.0
    # Unobserved echo (mask off) does not count.
    w_off = echo_weight(z, np.zeros_like(mask), dx_km=1.0, dy_km=1.0,
                        length_scale_km=3.0)
    assert not w_off.any()


def test_echo_noise_stays_near_echo_and_keeps_the_mean():
    from gpuwm.da.perturb import echo_weight

    rng = np.random.default_rng(3)
    priors = {m: {"u": rng.normal(5, 1, (NZ, NY, NX + 1)),
                  "v": rng.normal(2, 1, (NZ, NY + 1, NX)),
                  "thp": rng.normal(0, 1, (NZ, NY, NX)),
                  "qv": np.full((NZ, NY, NX), 8e-3),
                  "p": np.full((NZ, NY, NX), 9e4)} for m in range(4)}
    z = np.full((NZ, NY, NX), -10.0)
    z[2, 18:22, 20:24] = 40.0
    w = echo_weight(z, np.ones_like(z, bool), dx_km=1.0, dy_km=1.0,
                    length_scale_km=3.0)
    noise, record = additive_inflation(priors, _cfg(), seed=5, leg=2,
                                       scale=1.0, weight=w,
                                       stream="echo-noise")
    plain, _ = additive_inflation(priors, _cfg(), seed=5, leg=2, scale=1.0)
    for field in ("u", "v", "thp", "qv"):
        stack = np.stack([noise[m][field] for m in range(4)])
        np.testing.assert_allclose(stack.mean(axis=0), 0.0, atol=1e-12)
        assert np.abs(stack).max() > 0.0
    thp = np.stack([noise[m]["thp"] for m in range(4)])
    assert np.all(thp[:, :, 0, 0] == 0.0)          # far from the echo
    # Its own stream: not the domain-wide draw scaled.
    assert not np.allclose(noise[0]["thp"][:, 20, 22],
                           plain[0]["thp"][:, 20, 22])


def test_threaded_frame_draws_are_the_serial_tables_byte_for_byte(monkeypatch):
    """THE BREAKAGE: drawn one after another, a 9 km CONUS member's boundary
    frames cost 40 s of one host core at the start of every run while its
    card sat idle.  They are drawn on threads now; the tables must be the
    serial ones exactly, or every member's setup fingerprint moves."""
    import gpuwm.da.perturb as perturb

    base = _boundaries(frames=6)
    threaded, record = _member(base=base)
    monkeypatch.setattr(perturb, "_boundary_draws",
                        lambda draw, indices: {i: draw(i) for i in indices})
    serial, serial_record = _member(base=base)
    assert record["tables_sha256"] == serial_record["tables_sha256"]
    for a, b in zip(threaded.intervals, serial.intervals):
        for table in a.fields:
            for side in ("west", "east", "south", "north"):
                sa, sb = getattr(a.fields[table], side), getattr(b.fields[table], side)
                assert np.asarray(sa.value).tobytes() == np.asarray(sb.value).tobytes()
                assert np.asarray(sa.tendency).tobytes() == np.asarray(sb.tendency).tobytes()
