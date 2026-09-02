"""Device behaviour of the MPAS column-batch physics seam.

Runs the REAL chain (legacy RRTMG, revised-MO surface layer, Noah-MP,
YSU, KF/GF, WSM6) on synthetic moist columns and gates the contract
halves that need arrays: the read-only phase-1 guarantee (negative qv
included), the held-radiation cadence over 10 calls, bucket
accumulation, counter persistence, the identity-coupling proof, phase-2
in-place updates of all six species plus theta, and the restart
round-trip's bit-identical continuation.
"""

import datetime
import os
import subprocess
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpuwm.core import mpas_column_batch as mcb  # noqa: E402

NZ, NCOL = 20, 8
DT = 120.0
START = datetime.datetime(2021, 6, 1, 18, 0)


def _profile(seed=0):
    """A moist, slightly supersaturated warm sounding per column."""
    rng = np.random.default_rng(seed)
    z_iface = np.linspace(0.0, 16000.0, NZ + 1)
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    p_iface = 101325.0 * np.exp(-z_iface / 8000.0)
    p_mid = 101325.0 * np.exp(-z_mid / 8000.0)
    theta = 300.0 + 25.0 * z_mid / 16000.0
    exner = (p_mid / 1.0e5) ** (287.0 / 1004.0)
    temp = theta * exner
    rho_dry = p_mid / (287.0 * temp)
    qv = 0.014 * np.exp(-z_mid / 3000.0)
    qc = np.where(z_mid < 4000.0, 8.0e-4, 0.0)

    def cols(profile, jitter=1.0e-3):
        base = np.repeat(profile[:, None], NCOL, axis=1)
        base *= 1.0 + jitter * rng.standard_normal(base.shape)
        return np.ascontiguousarray(base, dtype=np.float32)

    fields = {
        "u": cols(np.full(NZ, 5.0)),
        "v": cols(np.full(NZ, -3.0)),
        "theta": cols(theta),
        "pressure": cols(p_mid, jitter=0.0),
        "pressure_interface": cols(p_iface, jitter=0.0),
        "z_interface": np.ascontiguousarray(
            np.repeat(z_iface[:, None], NCOL, axis=1), dtype=np.float32),
        "w": cols(0.05 * np.sin(np.linspace(0, np.pi, NZ + 1)),
                  jitter=0.0),
        "rho_dry": cols(rho_dry, jitter=0.0),
        "qv": cols(qv),
        "qc": cols(qc),
        "qr": cols(np.full(NZ, 1.0e-5)),
        "qi": np.zeros((NZ, NCOL), dtype=np.float32),
        "qs": np.zeros((NZ, NCOL), dtype=np.float32),
        "qg": np.zeros((NZ, NCOL), dtype=np.float32),
    }
    # THE NEGATIVE-QV LAW: the MPAS init carries small negative qv
    # values; the seam must clamp in scratch and never refuse.
    fields["qv"][-3:, 0] = -1.0e-7
    fields["qv"][-1, 1] = -5.0e-9
    return {name: cp.asarray(value) for name, value in fields.items()}


def _seam(cumulus_scheme="kf", cumulus_seconds=600.0):
    return mcb.run_mpas_column_batch(
        n_levels=NZ, n_columns=NCOL, dt=DT,
        radiation_seconds=600.0, surface_pbl_seconds=120.0,
        cumulus_seconds=cumulus_seconds, cumulus_scheme=cumulus_scheme,
        start_time=START,
        latitude_deg=np.full(NCOL, 35.0),
        longitude_deg=np.full(NCOL, -97.0),
        terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, NZ + 1),
        p_top_pa=float(101325.0 * np.exp(-2.0)), dx_m=15000.0)


def _phase2_state(inputs):
    """Caller-side post-transport arrays phase 2 updates in place."""
    return {name: inputs[name].copy()
            for name in ("theta", "qv", "qc", "qr", "qi", "qs", "qg",
                         "pressure", "rho_dry", "z_interface")}


def _step(seam, inputs, state2):
    result = seam.run_phase1(dt=DT, **inputs)
    receipt = seam.run_phase2(**state2)
    return result, receipt


@pytest.fixture(scope="module")
def kf_run():
    """One 10-step KF trajectory with captured per-step evidence."""
    seam = _seam()
    inputs = _profile()
    state2 = _phase2_state(inputs)
    snapshots = []
    for _ in range(10):
        result = seam.run_phase1(dt=DT, **inputs)
        held_lw = cp.asnumpy(seam._driver.rthratenlw).copy()
        held_sw = cp.asnumpy(seam._driver.rthratensw).copy()
        raw_pbl_dtheta = cp.asnumpy(
            seam._driver.last_ysu["dtheta"]).copy()
        pbl_stack_rtheta = cp.asnumpy(
            seam._driver.pbl_tendencies.rtheta).copy()
        receipt = seam.run_phase2(**state2)
        snapshots.append({
            "result": {
                name: cp.asnumpy(getattr(result, name)).copy()
                for name in mcb._OUTPUT_BUFFERS},
            "flags": (result.radiation_ran, result.surface_pbl_ran,
                      result.cumulus_ran),
            "held_lw": held_lw, "held_sw": held_sw,
            "raw_pbl_dtheta": raw_pbl_dtheta,
            "pbl_stack_rtheta": pbl_stack_rtheta,
            "rainncv": cp.asnumpy(receipt["rainncv"]).copy(),
            "buckets": {name: cp.asnumpy(value).copy() for name, value
                        in seam.accumulated_precipitation().items()},
            "counts": seam.call_counts,
        })
    return seam, inputs, state2, snapshots


def test_phase1_leaves_every_input_byte_identical():
    seam = _seam()
    inputs = _profile()
    before = {name: cp.asnumpy(value).tobytes()
              for name, value in inputs.items()}
    assert float(cp.asnumpy(inputs["qv"]).min()) < 0.0  # the law's case
    seam.run_phase1(dt=DT, **inputs)
    for name, value in inputs.items():
        assert cp.asnumpy(value).tobytes() == before[name], (
            f"phase 1 mutated caller array {name!r}")


def test_radiation_cadence_and_holds_over_ten_calls(kf_run):
    _, _, _, snapshots = kf_run
    flags = [snap["flags"][0] for snap in snapshots]
    assert flags == [True, False, False, False, False,
                     True, False, False, False, False]
    assert all(snap["flags"][1] for snap in snapshots)  # PBL every step
    # HELD tendencies: the stored radiation rates are byte-identical
    # across non-due steps and change at the due step.
    for index in (1, 2, 3, 4):
        assert snapshots[index]["held_lw"].tobytes() \
            == snapshots[0]["held_lw"].tobytes()
        assert snapshots[index]["held_sw"].tobytes() \
            == snapshots[0]["held_sw"].tobytes()
    assert snapshots[5]["held_sw"].tobytes() \
        != snapshots[4]["held_sw"].tobytes()


def test_kf_cadence_holds_cumulus_rates_between_due_calls(kf_run):
    _, _, _, snapshots = kf_run
    flags = [snap["flags"][2] for snap in snapshots]
    # cumulus_seconds=600 -> due at steps 1, 5, 10 (WRF MOD == 0 rule).
    assert flags == [True, False, False, False, True,
                     False, False, False, False, True]


def test_counters_advance_with_the_orchestration(kf_run):
    _, _, _, snapshots = kf_run
    counts = snapshots[-1]["counts"]
    assert counts["radiation"] == 2
    assert counts["ysu"] == 10
    assert counts["sfclay"] == 10
    assert counts["noah"] == 10
    assert counts["cumulus"] == 3


def test_identity_coupling_makes_held_stacks_raw(kf_run):
    """The PBL stack's mass-coupled rtheta is BITWISE the raw YSU rate.

    This is the proof (and the revert-check) of the chm == 1.0 identity
    construction: any change to c1h/c2h/mub2d/mup or the map factors on
    the column-batch state breaks byte equality here.
    """
    _, _, _, snapshots = kf_run
    for snap in snapshots:
        assert snap["pbl_stack_rtheta"].tobytes() \
            == snap["raw_pbl_dtheta"].tobytes()


def test_the_returned_tendencies_are_finite_and_complete(kf_run):
    _, _, _, snapshots = kf_run
    last = snapshots[-1]["result"]
    for name in mcb._OUTPUT_BUFFERS:
        assert np.isfinite(last[name]).all(), name
    assert last["dqg"].max() == 0.0 and last["dqg"].min() == 0.0
    assert np.abs(last["dtheta"]).max() > 0.0
    assert np.abs(last["du"]).max() > 0.0


def test_buckets_accumulate_across_phase2_calls(kf_run):
    _, _, _, snapshots = kf_run
    rainnc = [snap["buckets"]["RAINNC"] for snap in snapshots]
    for previous, current in zip(rainnc, rainnc[1:]):
        assert (current >= previous).all()
    assert rainnc[-1].max() > 0.0  # the supersaturated columns rain
    total = np.zeros(NCOL, dtype=np.float32)
    for snap in snapshots:
        total = total + np.maximum(snap["rainncv"], 0.0)
    np.testing.assert_allclose(rainnc[-1], total, rtol=1.0e-5)


def test_phase2_updates_all_six_species_and_theta_in_place(kf_run):
    _, inputs, state2, _ = kf_run
    changed = {name: cp.asnumpy(state2[name]).tobytes()
               != cp.asnumpy(inputs[name]).tobytes()
               for name in ("theta", "qv", "qc", "qr", "qi", "qs", "qg")}
    assert changed["theta"] and changed["qv"] and changed["qc"]
    assert changed["qr"]
    # Frozen categories grow from the seam's zero init once WSM6 runs a
    # cold upper troposphere.
    assert cp.asnumpy(state2["qi"]).max() > 0.0 or changed["qi"]
    assert cp.asnumpy(state2["qs"]).max() > 0.0 or changed["qs"]
    # Read-only phase-2 inputs stay byte-identical.
    for name in ("pressure", "rho_dry", "z_interface"):
        assert cp.asnumpy(state2[name]).tobytes() \
            == cp.asnumpy(inputs[name]).tobytes()


def test_strict_alternation_is_enforced(kf_run):
    seam, inputs, state2, _ = kf_run
    result = seam.run_phase1(dt=DT, **inputs)
    assert result is not None
    with pytest.raises(RuntimeError, match="alternation"):
        seam.run_phase1(dt=DT, **inputs)
    with pytest.raises(RuntimeError, match="step boundary"):
        seam.export_state()
    seam.run_phase2(**state2)  # restore the boundary for later tests


def test_wrong_dt_refuses(kf_run):
    seam, inputs, _, _ = kf_run
    with pytest.raises(ValueError, match="does not match"):
        seam.run_phase1(dt=60.0, **inputs)


def test_restart_round_trip_continues_bit_identically():
    inputs = _profile(seed=7)
    reference = _seam()
    ref_state2 = _phase2_state(inputs)
    for _ in range(3):
        _step(reference, inputs, ref_state2)
    payload = reference.export_state()
    resumed_state2 = {name: value.copy()
                      for name, value in ref_state2.items()}
    resumed = _seam()
    resumed.restore_state(payload)
    for step in range(4):
        ref_result, _ = _step(reference, inputs, ref_state2)
        res_result, _ = _step(resumed, inputs, resumed_state2)
        for name in mcb._OUTPUT_BUFFERS:
            assert cp.asnumpy(getattr(ref_result, name)).tobytes() \
                == cp.asnumpy(getattr(res_result, name)).tobytes(), (
                    f"{name} diverged on post-restore step {step}")
        for name in ("theta", "qv", "qc", "qr", "qi", "qs", "qg"):
            assert cp.asnumpy(ref_state2[name]).tobytes() \
                == cp.asnumpy(resumed_state2[name]).tobytes(), (
                    f"phase-2 {name} diverged on post-restore "
                    f"step {step}")
    assert resumed.step_index == reference.step_index
    assert resumed.call_counts == reference.call_counts


def test_restart_refuses_identity_and_coverage_mismatches():
    seam = _seam()
    inputs = _profile()
    _step(seam, inputs, _phase2_state(inputs))
    payload = seam.export_state()
    other = _seam(cumulus_scheme="kf", cumulus_seconds=1200.0)
    with pytest.raises(ValueError, match="identity"):
        other.restore_state(payload)
    broken = {**payload, "arrays": dict(payload["arrays"])}
    broken["arrays"].pop("driver/rthratenlw")
    fresh = _seam()
    with pytest.raises(ValueError, match="missing"):
        fresh.restore_state(broken)
    with pytest.raises(ValueError, match="exactly"):
        fresh.restore_state({"identity": payload["identity"]})


# ---------------------------------------------------------------------------
# Pool-purity law: phase-1 outputs are a function of the inputs alone,
# never of whatever a freed CuPy pool block happens to contain.
# ---------------------------------------------------------------------------

_POISON_WORDS = 64 * 1024 * 1024          # 256 MiB of float32


def _shallow_top_profile(seed=0):
    """_profile plus one column whose interface top sits at 4.2 hPa.

    MPAS hands the seam per-column tops that can sit far below the
    nominal p_top that sizes the LW Cavallo buffer march; this is the
    state that marched play nonpositive and let the LW band kernels
    index outside their coefficient tables.
    """
    inputs = _profile(seed)
    p_ifc = cp.asnumpy(inputs["pressure_interface"])
    p_mid = cp.asnumpy(inputs["pressure"])
    p_ifc[-1, 0] = 420.0                  # 4.2 hPa, still monotone
    p_mid[-1, 0] = 0.5 * (p_ifc[-2, 0] + p_ifc[-1, 0])
    inputs["pressure_interface"] = cp.asarray(p_ifc)
    inputs["pressure"] = cp.asarray(p_mid)
    return inputs


def _pool_poison_arm():
    """Subprocess arm: poison a large freed pool block, run phase 1.

    Invoked as ``python -c "... t._pool_poison_arm()" <pattern> <out>``.
    The block is filled with <pattern>, synchronized, and freed BEFORE
    seam construction, so every device allocation the phase-1 chain
    makes is carved out of poisoned history.
    """
    pattern, out_path = sys.argv[1], sys.argv[2]
    block = cp.empty(_POISON_WORDS, dtype=cp.float32)
    block.fill(np.float32(float(pattern)))
    cp.cuda.runtime.deviceSynchronize()
    del block                     # back to the pool, contents retained
    seam = _seam()
    inputs = _shallow_top_profile()
    result = seam.run_phase1(dt=DT, **inputs)
    payload = {name: cp.asnumpy(getattr(result, name))
               for name in mcb._OUTPUT_BUFFERS}
    payload["rthratenlw"] = cp.asnumpy(seam._driver.rthratenlw)
    payload["rthratensw"] = cp.asnumpy(seam._driver.rthratensw)
    np.savez(str(out_path), **payload)


def test_phase1_is_bitwise_invariant_to_freed_pool_contents(tmp_path):
    """Two fresh-process arms differing ONLY in the pattern (NaN vs 0.0)
    written into a large freed pool block before seam construction must
    produce bitwise-identical phase-1 outputs.  This is the restart
    purity law: a restore in a new process meets different pool history
    than the run that wrote the checkpoint, and the physics answer must
    not see the difference.
    """
    import gpuwm
    roots = [os.path.dirname(os.path.dirname(os.path.abspath(
        gpuwm.__file__))), os.path.dirname(os.path.abspath(__file__))]
    env = dict(os.environ)
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        roots + ([prior] if prior else []))
    arms = {}
    for name, pattern in (("nan", "nan"), ("zero", "0.0")):
        out = tmp_path / f"poison-{name}.npz"
        proc = subprocess.run(
            [sys.executable, "-c",
             "import test_mpas_column_batch_gpu as t; "
             "t._pool_poison_arm()", pattern, str(out)],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env, capture_output=True, text=True, timeout=3600)
        assert proc.returncode == 0, (
            f"poison arm {name!r} failed:\n{proc.stdout}\n{proc.stderr}")
        arms[name] = np.load(str(out))
    nan_arm, zero_arm = arms["nan"], arms["zero"]
    assert sorted(nan_arm.files) == sorted(zero_arm.files)
    for key in sorted(nan_arm.files):
        assert nan_arm[key].tobytes() == zero_arm[key].tobytes(), (
            f"phase-1 output {key!r} depends on freed-pool contents")


def test_grell_freitas_variant_runs_and_reports_every_step():
    seam = _seam(cumulus_scheme="gf", cumulus_seconds=DT)
    inputs = _profile(seed=3)
    state2 = _phase2_state(inputs)
    for _ in range(2):
        result, _ = _step(seam, inputs, state2)
        assert result.cumulus_ran
        for name in mcb._OUTPUT_BUFFERS:
            assert np.isfinite(
                cp.asnumpy(getattr(result, name))).all(), name
    assert seam.call_counts["cumulus"] == 2


# ---------------------------------------------------------------------------
# P3 arm (microphysics_scheme="p3", mp_physics=50; 2026-08-31).
# The same seam, the eight-scalar transport: qv/qc/qr/qi + ni/nr/qir/qib.
# ---------------------------------------------------------------------------

P3_SPECIES = ("qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib")


def _p3_profile(seed=0):
    """The WSM6 profile re-drawn onto P3's species set.

    qr carries a consistent rain number moment (~1e5 kg^-1 per 1e-5
    kg/kg, a plausible marine drop spectrum); ice starts cold at zero
    exactly as a DomainState mp=50 allocation does, so the first steps
    exercise P3's own nucleation rather than an invented ice state.
    """
    base = _profile(seed)
    inputs = {name: base[name] for name in
              ("u", "v", "theta", "pressure", "pressure_interface",
               "z_interface", "w", "rho_dry", "qv", "qc", "qr", "qi")}
    inputs["nr"] = cp.asarray(
        cp.asnumpy(base["qr"]) * np.float32(1.0e10), dtype=cp.float32)
    for name in ("ni", "qir", "qib"):
        inputs[name] = cp.zeros((NZ, NCOL), dtype=cp.float32)
    return inputs


def _p3_seam(cumulus_scheme="kf", cumulus_seconds=600.0):
    return mcb.run_mpas_column_batch(
        n_levels=NZ, n_columns=NCOL, dt=DT,
        microphysics_scheme="p3",
        radiation_seconds=600.0, surface_pbl_seconds=120.0,
        cumulus_seconds=cumulus_seconds, cumulus_scheme=cumulus_scheme,
        start_time=START,
        latitude_deg=np.full(NCOL, 35.0),
        longitude_deg=np.full(NCOL, -97.0),
        terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, NZ + 1),
        p_top_pa=float(101325.0 * np.exp(-2.0)), dx_m=15000.0)


def _p3_phase2_state(inputs):
    return {name: inputs[name].copy()
            for name in ("theta", "pressure", "z_interface") + P3_SPECIES}


def test_p3_phase1_leaves_every_input_byte_identical():
    seam = _p3_seam()
    inputs = _p3_profile()
    before = {name: cp.asnumpy(value).tobytes()
              for name, value in inputs.items()}
    assert float(cp.asnumpy(inputs["qv"]).min()) < 0.0  # the qv law
    result = seam.run_phase1(dt=DT, **inputs)
    for name, value in inputs.items():
        assert cp.asnumpy(value).tobytes() == before[name], (
            f"phase 1 mutated caller input {name}")
    for name in mcb._OUTPUT_BUFFERS:
        assert np.isfinite(cp.asnumpy(getattr(result, name))).all(), name
    # No phase-1 scheme forces snow or graupel on a P3 state.
    assert not cp.asnumpy(result.dqs).any()
    assert not cp.asnumpy(result.dqg).any()


def test_p3_phase2_updates_the_eight_species_and_theta_in_place():
    seam = _p3_seam()
    inputs = _p3_profile(seed=11)
    state2 = _p3_phase2_state(inputs)
    before = {name: cp.asnumpy(state2[name]).copy() for name in state2}
    receipts = []
    for step in range(3):
        seam.run_phase1(dt=DT, **inputs)
        receipts.append(seam.run_phase2(
            **state2, refl_10cm_due=(step == 2)))
    assert (cp.asnumpy(state2["theta"]) != before["theta"]).any()
    assert (cp.asnumpy(state2["qv"]) != before["qv"]).any()
    # The P3 receipt shape: five-slot scheme, no graupel key, and the
    # due step's own call computed REFL_10CM (WRF diagflag semantics).
    for receipt in receipts:
        assert "graupelncv" not in receipt
    assert set(receipts[0]) == {"rainncv", "snowncv", "sr"}
    assert set(receipts[2]) == {"rainncv", "snowncv", "sr", "refl_10cm"}
    assert receipts[2]["refl_10cm"].shape == (NZ, NCOL)
    buckets = seam.accumulated_precipitation()
    assert set(buckets) == {"RAINNC", "SNOWNC", "RAINC"}, (
        "P3 has no graupel accumulator (physics_inventory mp=50 row)")


def test_p3_species_direction_refusals():
    seam = _p3_seam()
    inputs = _p3_profile()
    with pytest.raises(ValueError, match="qs"):
        seam.run_phase1(dt=DT, **inputs,
                        qs=cp.zeros((NZ, NCOL), dtype=cp.float32))
    incomplete = {name: value for name, value in inputs.items()
                  if name != "qir"}
    with pytest.raises(ValueError, match="qir"):
        seam.run_phase1(dt=DT, **incomplete)
    # Phase 2: rho_dry is WSM6 state, refused on a P3 seam.
    seam.run_phase1(dt=DT, **inputs)
    state2 = _p3_phase2_state(inputs)
    with pytest.raises(ValueError, match="rho_dry"):
        seam.run_phase2(**state2, rho_dry=inputs["rho_dry"])
    seam.run_phase2(**state2)   # leave the seam at a clean boundary


def test_p3_restart_round_trip_continues_bit_identically():
    """THE CARD GATE: a restored P3 seam continues bit-identically.

    Covers the P3 additions end to end: th_old/qv_old (the cross-step
    supersaturation carriers) and the P3 radii ride the payload; the
    five-slot buckets restore; the held tendencies keep returning.
    """
    inputs = _p3_profile(seed=7)
    reference = _p3_seam()
    ref_state2 = _p3_phase2_state(inputs)
    for _ in range(3):
        reference.run_phase1(dt=DT, **inputs)
        reference.run_phase2(**ref_state2)
    payload = reference.export_state()
    assert payload["identity"]["microphysics"] == "p3"
    for name in ("state/th_old", "state/qv_old"):
        assert name in payload["arrays"], (
            f"{name} missing: a resumed run would repeat P3's "
            "first-step supersaturation transient")
    assert "state/effs" not in payload["arrays"]
    resumed_state2 = {name: value.copy()
                      for name, value in ref_state2.items()}
    resumed = _p3_seam()
    resumed.restore_state(payload)
    for step in range(4):
        ref_result = reference.run_phase1(dt=DT, **inputs)
        res_result = resumed.run_phase1(dt=DT, **inputs)
        reference.run_phase2(**ref_state2)
        resumed.run_phase2(**resumed_state2)
        for name in mcb._OUTPUT_BUFFERS:
            assert cp.asnumpy(getattr(ref_result, name)).tobytes() \
                == cp.asnumpy(getattr(res_result, name)).tobytes(), (
                    f"{name} diverged on post-restore step {step}")
        for name in ("theta",) + P3_SPECIES:
            assert cp.asnumpy(ref_state2[name]).tobytes() \
                == cp.asnumpy(resumed_state2[name]).tobytes(), (
                    f"phase-2 {name} diverged on post-restore "
                    f"step {step}")
    assert resumed.step_index == reference.step_index
    assert resumed.call_counts == reference.call_counts


def test_p3_and_wsm6_payloads_refuse_each_other():
    p3_seam = _p3_seam()
    p3_inputs = _p3_profile()
    p3_seam.run_phase1(dt=DT, **p3_inputs)
    p3_seam.run_phase2(**_p3_phase2_state(p3_inputs))
    p3_payload = p3_seam.export_state()
    wsm6_seam = _seam()
    wsm6_inputs = _profile()
    wsm6_seam.run_phase1(dt=DT, **wsm6_inputs)
    wsm6_seam.run_phase2(**_phase2_state(wsm6_inputs))
    wsm6_payload = wsm6_seam.export_state()
    assert "microphysics" not in wsm6_payload["identity"], (
        "the wsm6 identity must stay byte-identical to pre-P3 schema "
        "v2 so stored payloads keep restoring")
    with pytest.raises(ValueError, match="identity"):
        _p3_seam().restore_state(wsm6_payload)
    with pytest.raises(ValueError, match="identity"):
        _seam().restore_state(p3_payload)


# ---------------------------------------------------------------------------
# Aerosol-aware Thompson arm (microphysics_scheme="thompson_aero",
# mp_physics=28; 2026-09-01).  The same seam, the eleven-scalar transport:
# qv/qc/qr/qi/qs/qg + ni/nr/nc + nwfa/nifa.
# ---------------------------------------------------------------------------

MP28_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg",
                "ni", "nr", "nc", "nwfa", "nifa")


def _mp28_profile(seed=0, caller_aerosol=False):
    """The WSM6 profile re-drawn onto mp=28's species set.

    qr carries a consistent rain number moment (as the P3 profile
    does); ice number, droplet number and both aerosol tracers start at
    zero exactly as a DomainState mp=28 allocation does -- nc is
    bootstrapped by the scheme's terminal rediagnosis and the aerosol
    by the seam's first-contact thompson_init.  ``caller_aerosol=True``
    is the route that brings its own aerosol: uniform per-kg number
    fields far above thompson_init's presence threshold.
    """
    base = _profile(seed)
    inputs = {name: base[name] for name in
              ("u", "v", "theta", "pressure", "pressure_interface",
               "z_interface", "w", "rho_dry", "qv", "qc", "qr", "qi",
               "qs", "qg")}
    inputs["nr"] = cp.asarray(
        cp.asnumpy(base["qr"]) * np.float32(1.0e10), dtype=cp.float32)
    for name in ("ni", "nc", "nwfa", "nifa"):
        inputs[name] = cp.zeros((NZ, NCOL), dtype=cp.float32)
    if caller_aerosol:
        inputs["nwfa"] = cp.full((NZ, NCOL), 2.0e8, dtype=cp.float32)
        inputs["nifa"] = cp.full((NZ, NCOL), 1.0e6, dtype=cp.float32)
    return inputs


def _mp28_seam(cumulus_scheme="kf", cumulus_seconds=600.0):
    return mcb.run_mpas_column_batch(
        n_levels=NZ, n_columns=NCOL, dt=DT,
        microphysics_scheme="thompson_aero",
        radiation_seconds=600.0, surface_pbl_seconds=120.0,
        cumulus_seconds=cumulus_seconds, cumulus_scheme=cumulus_scheme,
        start_time=START,
        latitude_deg=np.full(NCOL, 35.0),
        longitude_deg=np.full(NCOL, -97.0),
        terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, NZ + 1),
        p_top_pa=float(101325.0 * np.exp(-2.0)), dx_m=15000.0)


def _mp28_phase2_state(inputs):
    return {name: inputs[name].copy()
            for name in ("theta", "pressure", "z_interface") + MP28_SPECIES}


def _nwfa2d_from_the_synthetic_fill(z_interface):
    """thompson_init:500-510 on the caller's geometry, in float64.

    The instrument for the synthetic branch: ``hgt`` is z8w[:nz] (the
    port's stated height field), so hgt(1) is the terrain interface and
    hgt(2) the first interface above it; the test columns sit at sea
    level, so ``h_01 = 0.8``.
    """
    from gpuwm.core.thompson_aerosol_contract import (
        NA_CCN0, NA_CCN1, NWFA2D_EMISSION_COEFF,
        NWFA2D_REFERENCE_DEPTH_M, PROFILE_H01_LOW)
    z = np.asarray(cp.asnumpy(z_interface), dtype=np.float64)
    hgt1, hgt2 = z[0], z[1]
    assert (hgt1 <= 1000.0).all()
    ni_ccn3 = -np.log(NA_CCN1 / NA_CCN0) / PROFILE_H01_LOW
    nwfa1 = NA_CCN1 + NA_CCN0 * np.exp(-((hgt2 - hgt1) / 1000.0) * ni_ccn3)
    z1 = hgt2 - hgt1
    return nwfa1 * NWFA2D_EMISSION_COEFF * (NWFA2D_REFERENCE_DEPTH_M / z1)


def test_mp28_phase1_leaves_every_input_byte_identical():
    seam = _mp28_seam()
    inputs = _mp28_profile()
    before = {name: cp.asnumpy(value).tobytes()
              for name, value in inputs.items()}
    assert float(cp.asnumpy(inputs["qv"]).min()) < 0.0  # the qv law
    result = seam.run_phase1(dt=DT, **inputs)
    for name, value in inputs.items():
        assert cp.asnumpy(value).tobytes() == before[name], (
            f"phase 1 mutated caller input {name}")
    for name in mcb._OUTPUT_BUFFERS:
        assert np.isfinite(cp.asnumpy(getattr(result, name))).all(), name
    # First contact is phase 2's: phase 1 is read-only and never
    # touches the aerosol fields on this seam.
    assert seam.aerosol_init_receipt is None


def test_mp28_first_contact_seeds_the_synthetic_aerosol_in_the_callers_arrays():
    """A caller with no aerosol data gets WRF's thompson_init, once.

    Both presence tests fail on all-zero fields, both synthetic fills
    run INTO the caller's arrays with the caller's own z_interface, and
    the surface emission is thompson_init:509-510 on that geometry --
    the value a nominal-mesh derivation could only match by accident.
    """
    seam = _mp28_seam()
    inputs = _mp28_profile(seed=3)
    state2 = _mp28_phase2_state(inputs)
    assert not cp.asnumpy(state2["nwfa"]).any()
    assert not cp.asnumpy(state2["nifa"]).any()
    contacts = []
    first_contact = seam._thompson_first_contact
    seam._thompson_first_contact = (
        lambda: contacts.append(1) or first_contact())
    seam.run_phase1(dt=DT, **inputs)
    seam.run_phase2(**state2)
    assert seam.aerosol_init_receipt == {
        "ccn_filled": True, "in_filled": True,
        "nwfa2d_source": "thompson_init"}
    assert (cp.asnumpy(state2["nwfa"]) > 0.0).all()
    assert (cp.asnumpy(state2["nifa"]) > 0.0).all()
    expected = _nwfa2d_from_the_synthetic_fill(inputs["z_interface"])
    got = cp.asnumpy(seam._state.nwfa2d).reshape(NCOL)
    np.testing.assert_allclose(got, expected, rtol=2.0e-6)
    assert not cp.asnumpy(seam._state.nifa2d).any(), (
        "WRF never derives nifa2d; it stays exactly zero")
    # Exactly once per constructed seam: the second step does not
    # re-test or re-fill the evolving field.
    seam.run_phase1(dt=DT, **inputs)
    seam.run_phase2(**state2)
    assert contacts == [1]


def test_mp28_caller_aerosol_takes_the_has_ccn_branch():
    """A caller that brings aerosol keeps it, and gets real.exe's nwfa2d.

    thompson_init's presence tests pass, no fill runs (the caller's
    fields are untouched by anything but the scheme), and the surface
    emission is module_initialize_real.F:4518-4519 on the caller's
    lowest level and lowest-layer depth, single precision, bit for bit.
    """
    from gpuwm.core import constants
    seam = _mp28_seam()
    inputs = _mp28_profile(seed=5, caller_aerosol=True)
    state2 = _mp28_phase2_state(inputs)
    nwfa_in = cp.asnumpy(state2["nwfa"]).copy()
    seam.run_phase1(dt=DT, **inputs)
    seam.run_phase2(**state2)
    assert seam.aerosol_init_receipt == {
        "ccn_filled": False, "in_filled": False,
        "nwfa2d_source": "caller-aerosol"}
    g = np.float32(constants.G)
    z = cp.asnumpy(inputs["z_interface"]).astype(np.float32)
    z1 = (z[1] * g - z[0] * g) / g
    expected = ((nwfa_in[0] * np.float32(0.000196))
                * (np.float32(50.0) / z1)).astype(np.float32)
    got = cp.asnumpy(seam._state.nwfa2d).reshape(NCOL)
    assert got.tobytes() == expected.tobytes()
    assert not cp.asnumpy(seam._state.nifa2d).any()


def test_mp28_phase2_updates_the_eleven_species_and_theta_in_place():
    seam = _mp28_seam()
    inputs = _mp28_profile(seed=11)
    state2 = _mp28_phase2_state(inputs)
    before = {name: cp.asnumpy(state2[name]).copy() for name in state2}
    receipts = []
    for step in range(3):
        seam.run_phase1(dt=DT, **inputs)
        receipts.append(seam.run_phase2(
            **state2, refl_10cm_due=(step == 2)))
    assert (cp.asnumpy(state2["theta"]) != before["theta"]).any()
    assert (cp.asnumpy(state2["qv"]) != before["qv"]).any()
    # The prognostic droplet number bootstraps from zero where cloud
    # exists (terminal rediagnosis), and the aerosol evolves under the
    # scheme after first contact seeded it.
    assert (cp.asnumpy(state2["nc"]) != before["nc"]).any()
    assert (cp.asnumpy(state2["nwfa"]) != before["nwfa"]).any()
    # The mp=28 receipt shape: seven-slot scheme with a graupel key,
    # and the due step's own call computed REFL_10CM.
    for receipt in receipts:
        assert "graupelncv" in receipt
    assert set(receipts[0]) == {"rainncv", "snowncv", "graupelncv", "sr"}
    assert set(receipts[2]) == {"rainncv", "snowncv", "graupelncv", "sr",
                                "refl_10cm"}
    assert receipts[2]["refl_10cm"].shape == (NZ, NCOL)
    buckets = seam.accumulated_precipitation()
    assert set(buckets) == {"RAINNC", "SNOWNC", "GRAUPELNC", "RAINC"}, (
        "mp=28's driver arm binds GRAUPELNC (module_microphysics_driver"
        ".F:1089)")


def test_mp28_species_direction_refusals():
    seam = _mp28_seam()
    inputs = _mp28_profile()
    with pytest.raises(ValueError, match="qir"):
        seam.run_phase1(dt=DT, **inputs,
                        qir=cp.zeros((NZ, NCOL), dtype=cp.float32))
    for absent in ("nc", "nwfa"):
        incomplete = {name: value for name, value in inputs.items()
                      if name != absent}
        with pytest.raises(ValueError, match=rf"\['{absent}'\]"):
            seam.run_phase1(dt=DT, **incomplete)
    # Phase 2: rho_dry is WSM6 state, refused on an mp=28 seam.
    seam.run_phase1(dt=DT, **inputs)
    state2 = _mp28_phase2_state(inputs)
    with pytest.raises(ValueError, match="rho_dry"):
        seam.run_phase2(**state2, rho_dry=inputs["rho_dry"])
    seam.run_phase2(**state2)   # leave the seam at a clean boundary


def test_mp28_restart_round_trip_continues_bit_identically():
    """THE CARD GATE: a restored mp=28 seam continues bit-identically.

    Covers the mp=28 additions end to end: the surface emission pair
    nwfa2d/nifa2d and the WSM6 radius triple ride the payload, the
    first-contact receipt rides the scalars so the resumed seam never
    re-fills, the seven-slot buckets restore, the held tendencies keep
    returning, and every transported species stays byte-equal.
    """
    inputs = _mp28_profile(seed=7)
    reference = _mp28_seam()
    ref_state2 = _mp28_phase2_state(inputs)
    for _ in range(3):
        reference.run_phase1(dt=DT, **inputs)
        reference.run_phase2(**ref_state2)
    payload = reference.export_state()
    assert payload["identity"]["microphysics"] == "thompson_aero"
    for name in ("state/nwfa2d", "state/nifa2d", "state/effc",
                 "state/effi", "state/effs", "state/h_diabatic"):
        assert name in payload["arrays"], (
            f"{name} missing: a resumed run would lose mp=28 state "
            "nothing can rebuild from the caller's species")
    assert payload["arrays"]["state/nwfa2d"].any()
    assert payload["scalars"]["aerosol_init"] == \
        reference.aerosol_init_receipt
    resumed_state2 = {name: value.copy()
                      for name, value in ref_state2.items()}
    resumed = _mp28_seam()
    assert resumed.aerosol_init_receipt is None
    resumed.restore_state(payload)
    assert resumed.aerosol_init_receipt == reference.aerosol_init_receipt
    assert cp.asnumpy(resumed._state.nwfa2d).tobytes() == \
        cp.asnumpy(reference._state.nwfa2d).tobytes()
    for step in range(4):
        ref_result = reference.run_phase1(dt=DT, **inputs)
        res_result = resumed.run_phase1(dt=DT, **inputs)
        reference.run_phase2(**ref_state2)
        resumed.run_phase2(**resumed_state2)
        for name in mcb._OUTPUT_BUFFERS:
            assert cp.asnumpy(getattr(ref_result, name)).tobytes() \
                == cp.asnumpy(getattr(res_result, name)).tobytes(), (
                    f"{name} diverged on post-restore step {step}")
        for name in ("theta",) + MP28_SPECIES:
            assert cp.asnumpy(ref_state2[name]).tobytes() \
                == cp.asnumpy(resumed_state2[name]).tobytes(), (
                    f"phase-2 {name} diverged on post-restore "
                    f"step {step}")
    assert resumed.step_index == reference.step_index
    assert resumed.call_counts == reference.call_counts
    for name, value in reference.accumulated_precipitation().items():
        assert cp.asnumpy(value).tobytes() == cp.asnumpy(
            resumed.accumulated_precipitation()[name]).tobytes(), name


def test_mp28_restore_refuses_a_payload_without_the_first_contact_record():
    seam = _mp28_seam()
    inputs = _mp28_profile()
    seam.run_phase1(dt=DT, **inputs)
    seam.run_phase2(**_mp28_phase2_state(inputs))
    payload = seam.export_state()
    del payload["scalars"]["aerosol_init"]
    with pytest.raises(ValueError, match="aerosol_init"):
        _mp28_seam().restore_state(payload)


def test_mp28_payloads_refuse_the_other_schemes():
    mp28 = _mp28_seam()
    mp28_inputs = _mp28_profile()
    mp28.run_phase1(dt=DT, **mp28_inputs)
    mp28.run_phase2(**_mp28_phase2_state(mp28_inputs))
    mp28_payload = mp28.export_state()
    wsm6 = _seam()
    wsm6_inputs = _profile()
    wsm6.run_phase1(dt=DT, **wsm6_inputs)
    wsm6.run_phase2(**_phase2_state(wsm6_inputs))
    wsm6_payload = wsm6.export_state()
    p3 = _p3_seam()
    p3_inputs = _p3_profile()
    p3.run_phase1(dt=DT, **p3_inputs)
    p3.run_phase2(**_p3_phase2_state(p3_inputs))
    p3_payload = p3.export_state()
    for foreign in (wsm6_payload, p3_payload):
        with pytest.raises(ValueError, match="identity"):
            _mp28_seam().restore_state(foreign)
    with pytest.raises(ValueError, match="identity"):
        _seam().restore_state(mp28_payload)
    with pytest.raises(ValueError, match="identity"):
        _p3_seam().restore_state(mp28_payload)


def test_wsm6_and_p3_payload_layouts_are_unchanged_by_the_mp28_arm():
    """The pre-mp28 layouts, key for key.

    The identity key set, the scalars key set and the state-manifest
    key set of a wsm6 and a p3 payload are exactly what they were
    before the mp=28 arm existed: no aerosol receipt, no emission pair,
    no new identity key.  A stored payload from either scheme keeps
    restoring.
    """
    scalar_keys = {"step_index", "elapsed_seconds", "call_counts",
                   "microphysics_updates", "ysu_nan_guard_fires",
                   "kf_history_time", "carriers"}
    identity_keys = {"schema", "n_levels", "n_columns", "dt", "stepra",
                     "stepbl", "stepcu", "cumulus_scheme", "start_time",
                     "p_top_pa", "dx_m", "gf_ishallow", "gf_dx_column",
                     "wsm6_hail_opt", "xice_threshold", "xland_source"}
    cases = (
        (_seam, _profile, _phase2_state, identity_keys,
         {"state/effc", "state/effi", "state/effs", "state/h_diabatic"}),
        (_p3_seam, _p3_profile, _p3_phase2_state,
         identity_keys | {"microphysics"},
         {"state/effc", "state/effi", "state/h_diabatic",
          "state/th_old", "state/qv_old"}),
    )
    for factory, profile, phase2, expected_identity, expected_state in cases:
        seam = factory()
        inputs = profile()
        seam.run_phase1(dt=DT, **inputs)
        seam.run_phase2(**phase2(inputs))
        payload = seam.export_state()
        assert set(payload["identity"]) == expected_identity
        assert set(payload["scalars"]) == scalar_keys
        assert {key for key in payload["arrays"]
                if key.startswith("state/")} == expected_state
        assert seam.aerosol_init_receipt is None
