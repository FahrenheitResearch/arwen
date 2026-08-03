"""GABLS1: the registry contract, the transcribed authority, and the run.

Unlike the other case tests this one DOES integrate the case, because
integrating it is a CPU single-column job of a few seconds -- that is
the whole reason the stable-limb calibration target could be adopted at
all.  What the run asserts is the published score, not a golden.
"""
from __future__ import annotations

import math

import numpy as np

from gpuwm.verify.cases import gabls1


def test_the_case_registry_discovers_it():
    from gpuwm.verify import cases

    manifest = {entry["name"]: entry for entry in cases.manifest()}
    assert "gabls1" in manifest, sorted(manifest)
    assert {"verify", "script"} <= set(
        manifest["gabls1"]["capabilities"])


def test_nan_is_reported_but_never_a_gate_row():
    assert "nan" not in gabls1.GATES
    for name, (lo, hi) in gabls1.GATES.items():
        assert lo < hi, name


def test_the_prescribed_setup_is_the_published_one():
    """Transcription check on the parts a typo would silently corrupt.

    Each assertion restates a number from the case description in a
    DIFFERENT form from the constant that carries it, so a copy of the
    constant into the test could not pass by itself.
    """
    dz, z = gabls1.grid()
    assert dz == 6.25 and z.size == 64 and z[-1] + 0.5 * dz == 400.0
    # theta: 265 K flat below 100 m, 268 K at the 400 m top
    th = gabls1.sounding(z)
    assert np.all(th[z <= 100.0] == 265.0)
    np.testing.assert_allclose(
        gabls1.sounding(np.array([400.0]))[0], 268.0)
    # nine hours of cooling at 0.25 K/h is 2.25 K
    np.testing.assert_allclose(
        gabls1.COOLING_RATE_K_PER_HOUR * gabls1.RUN_SECONDS / 3600.0,
        2.25)
    # f at 73 N.  The case description prescribes the ROUNDED
    # 1.39e-4 s^-1; 2*Omega*sin(73 deg) is 1.39467e-4, 0.34% higher.
    # The prescribed value is what every participating model ran, so it
    # is what is registered -- this check only confirms which latitude
    # it came from.
    np.testing.assert_allclose(
        gabls1.FCOR, 2.0 * 7.292e-5 * math.sin(math.radians(73.0)),
        rtol=5e-3)
    # initial TKE 0.4 at the surface, zero at and above 250 m
    e0 = gabls1.initial_energy(z)
    assert e0[0] > 0.35 and np.all(e0[z >= 250.0] == 0.0)
    # the depth definition is Beare's 5%/0.95, not something else
    assert (gabls1.DEPTH_STRESS_FRACTION,
            gabls1.DEPTH_EXTRAPOLATION) == (0.05, 0.95)


def test_the_gates_are_the_published_ensemble_and_nothing_narrower():
    """The depth gate must be the SINGLE-COLUMN span, not the LES band.

    This case scores a single-column closure, so its peer group is the
    nineteen single-column models, which span 120-483 m.  Gating on the
    LES 177 +/- 16 m would be scoring a closure against an instrument it
    is not -- and would make a result in the middle of the operational
    pack indistinguishable from a broken column.
    """
    assert gabls1.GATES["boundary_layer_depth"] == gabls1.SCM_DEPTH_SPAN_M
    lo, hi = gabls1.GATES["depth_les_ratio"]
    ref = gabls1.LES_REFERENCE["boundary_layer_depth_m"][0]
    np.testing.assert_allclose((lo, hi),
                               (gabls1.SCM_DEPTH_SPAN_M[0] / ref,
                                gabls1.SCM_DEPTH_SPAN_M[1] / ref),
                               rtol=2e-3)
    # the published spread must sit inside every band it is the basis of
    for key, (mean, sigma) in (
            ("friction_velocity",
             gabls1.LES_REFERENCE["friction_velocity"]),
            ("surface_heat_flux",
             gabls1.LES_REFERENCE["surface_heat_flux_kinematic"]),
            ("surface_wind_angle",
             gabls1.LES_REFERENCE["surface_wind_angle_deg"])):
        band = gabls1.GATES[key]
        assert band[0] <= mean - 2.0 * sigma, key
        assert mean + 2.0 * sigma <= band[1], key


def test_the_closure_scores_inside_the_large_eddy_band():
    """THE RESULT, and the reason this case is in the tree.

    Run as prescribed, the closure's stable limb lands inside ONE
    published standard deviation of the large-eddy reference on every
    quantity the intercomparison tabulates -- against a single-column
    ensemble that spans a factor of four and whose mean is 61% too deep:

        quantity            LES (Cuxart Tab. IV)   19 SCMs      SASE
        depth h [m]              177 +/- 16       285 +/- 108   185.3
        u* [m/s]                0.29 +/- 0.02     0.30 +/- 0.03  0.2998
        w'theta'_s [K m/s]    -0.012 +/- 0.002  -0.015 +/- 0.005 -0.0122
        cross-isobar angle       35 +/- 3         33 +/- 7       36.1

    with a super-geostrophic nocturnal jet of 9.85 m/s at 184 m -- the
    structure Cuxart et al. report the over-mixing operational schemes
    fail to produce at all.

    THIS IS A SCORE, NOT A CALIBRATION.  No constant in the closure was
    moved to obtain it and no bound here was chosen from it: the bands
    asserted are the published ensemble's, set in the case module beside
    their citations.  The assertions below are deliberately looser than
    the measured values so that this test reports a REGIME and not a
    trajectory -- one sigma of the LES reference on the depth, which is
    the claim, and the gate bands on the rest.
    """
    m = gabls1.run()
    assert m["nan"] == 0.0
    for name, (lo, hi) in gabls1.GATES.items():
        assert lo <= m[name] <= hi, (name, m[name])
    ref, sigma = gabls1.LES_REFERENCE["boundary_layer_depth_m"]
    assert abs(m["boundary_layer_depth"] - ref) <= sigma, m[
        "boundary_layer_depth"]
    # the jet exists and is super-geostrophic, near the top of the layer
    assert m["jet_max_speed"] > gabls1.GEO_U
    assert 0.7 <= m["jet_height"] / m["boundary_layer_depth"] <= 1.3


def _rescored(c_e, c_ks, nz=32):
    """Score the case with the two coupled constants moved together.

    Mutating module constants is the only way to sweep a REGISTERED
    value without registering a second one, and it is restored in a
    finally.  ``C_KV = C_E**(1/3)`` follows because the log law forces
    it -- sweeping C_E without it would be sweeping a broken closure.
    """
    from gpuwm.verify import sase_ref as R

    saved = (R.C_E, R.C_KV, R.C_KS)
    try:
        R.C_E, R.C_KV, R.C_KS = c_e, c_e ** (1.0 / 3.0), c_ks
        return gabls1.run(nz=nz)
    finally:
        R.C_E, R.C_KV, R.C_KS = saved


def test_the_benchmark_refuses_the_tke_deficit_move():
    """THE C_E QUESTION, SETTLED -- against a published target, which is
    what makes this calibration and not fitting.

    The registered TKE deficit (e/u*^2 ~ 1 against an observed 3.3-5.5)
    has stood OPEN because closing it needs a joint (C_E, C_KS) move and
    no stable-limb target existed to move against.  One exists now.  It
    says no, three times over:

    * C_E ALONE is refused outright -- at the Mellor-Yamada 0.1704 the
      depth is EIGHT published sigma too deep, in the operational pack
      Cuxart et al. diagnose as over-mixing, with that pack's own
      cross-isobar signature.
    * THE DEFICIT IS REAL: at the registered constants the peak subgrid
      energy is 0.0816 m2/s2 against an LES ~0.3 (participating models
      0.2-0.5) -- 3.7x low, independently reproducing the registered
      claim against a published number for the first time.
    * THE JOINT MOVE is the right SHAPE (scaling C_KS with C_E holds the
      depth in band, exactly as the (C_r/C_E)^2 algebra predicts) and is
      still not worth taking: depth degrades monotonically while energy
      improves monotonically, and the energy never reaches its own
      observed band before the depth has spent most of its margin.

    Every number is in the C_E constant's own registry block.
    """
    ref, sigma = gabls1.LES_REFERENCE["boundary_layer_depth_m"]
    # The SWEEP runs at nz = 32, which the case's own grid() docstring
    # records as converged to 0.3% -- a verdict about a 6-sigma miss
    # cannot turn on a 0.3% grid effect, and the headline score above
    # is quoted at the prescribed nz = 64.
    head = gabls1.run(nz=32)
    # (1) C_E alone: refused, and by an amount no one can round away
    alone = _rescored(0.1704, 0.25)
    assert alone["boundary_layer_depth"] > ref + 6.0 * sigma, alone[
        "boundary_layer_depth"]
    assert alone["surface_wind_angle"] < 30.0      # the over-mixing pack's
    # (2) the deficit is real against the published LES energy level
    assert head["peak_energy"] < 0.5 * 0.3, head["peak_energy"]
    # (3) the joint move: right shape, monotone trade, not worth taking
    joint = [_rescored(c_e, 0.25 * c_e / 0.93)
             for c_e in (0.60, 0.40, 0.2734)]
    depths = [head["boundary_layer_depth"]] + [
        m["boundary_layer_depth"] for m in joint]
    energies = [head["peak_energy"]] + [m["peak_energy"] for m in joint]
    for m in joint:                                # the shape is right
        assert abs(m["boundary_layer_depth"] - ref) <= 2.0 * sigma, m[
            "boundary_layer_depth"]
    assert all(b > a for a, b in zip(depths, depths[1:])), depths
    assert all(b > a for a, b in zip(energies, energies[1:])), energies
    # ... and it still does not close the deficit it was for
    assert energies[-1] < 0.3, energies[-1]
    # the registered pair is the best of them on the scored quantity
    assert min(abs(d - ref) for d in depths) == abs(depths[0] - ref)


def test_the_rejected_stable_coefficient_over_deepens_it_like_the_pack():
    """INDEPENDENT CORROBORATION of a RED this project already carries.

    ``RunConfig.sase_stable_dissipation`` ships False because it exits
    the closure's own observation-band gate on a real frame.  That is
    one instrument on one case.  Here a SECOND, published instrument
    says the same thing from a different direction: with that switch on,
    the depth goes 185 -> 287 m, which is 62% above the large-eddy
    reference and lands the closure squarely in the operational
    first-order pack (396 +/- 60 m is where Cuxart et al. put the
    schemes they diagnose as mixing too much below the inversion, and
    287 is inside its lower half).  The surface heat flux nearly doubles
    and the cross-isobar angle falls toward the over-mixing group's own
    28 +/- 6 degrees.

    Two independent instruments, same verdict, opposite construction.
    That is what adopting a published benchmark buys.
    """
    head = gabls1.run()
    red = gabls1.run(stable_dissipation=True)
    ref = gabls1.LES_REFERENCE["boundary_layer_depth_m"][0]
    assert red["boundary_layer_depth"] > head["boundary_layer_depth"]
    assert red["boundary_layer_depth"] > 1.4 * ref
    assert abs(red["surface_heat_flux"]) > 1.5 * abs(
        head["surface_heat_flux"])
    # and the additive channel moves it the OTHER way, staying in band
    add = gabls1.run(additive_dissipation=True)
    assert add["boundary_layer_depth"] < head["boundary_layer_depth"]
    sigma = gabls1.LES_REFERENCE["boundary_layer_depth_m"][1]
    assert abs(add["boundary_layer_depth"] - ref) <= sigma, add[
        "boundary_layer_depth"]
