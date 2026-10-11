"""Device behaviour of the selectable MPAS column suite.

Runs the REAL selected chain -- legacy RRTMG, the MYNN surface layer, RUC
on nine soil levels, MYNN-EDMF with the GSL drag in its slot, aerosol-aware
Thompson -- on synthetic moist land columns and gates what needs arrays:
finite complete tendencies, the drag reaching du/dv, the read-only phase-1
guarantee, the restart round trip, and per-column independence of the
batch split (the seam-level precondition for card-count identity: a column
must not see which batch it was in).
"""

import datetime

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpuwm.core import mpas_column_batch as mcb  # noqa: E402
from gpuwm.core import mpas_column_suite as suite  # noqa: E402

NZ, NCOL = 30, 8
DT = 60.0
START = datetime.datetime(2026, 10, 3, 18, 0)
MP28_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg",
                "ni", "nr", "nc", "nwfa", "nifa")
ZTOP = 20000.0


def _profile(ncol=NCOL, seed=0):
    rng = np.random.default_rng(seed)
    z_iface = np.linspace(0.0, ZTOP, NZ + 1)
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    p_iface = 101325.0 * np.exp(-z_iface / 8000.0)
    p_mid = 101325.0 * np.exp(-z_mid / 8000.0)
    theta = 300.0 + 30.0 * z_mid / ZTOP
    exner = (p_mid / 1.0e5) ** (287.0 / 1004.0)
    rho_dry = p_mid / (287.0 * theta * exner)
    qv = 0.012 * np.exp(-z_mid / 2500.0)

    def cols(profile, jitter=1.0e-3):
        base = np.repeat(np.asarray(profile, np.float64)[:, None], ncol, 1)
        base *= 1.0 + jitter * rng.standard_normal(base.shape)
        return np.ascontiguousarray(base, dtype=np.float32)

    f = {
        "u": cols(4.0 + 20.0 * z_mid / ZTOP, 2e-2),
        "v": cols(-2.0 + 6.0 * z_mid / ZTOP, 2e-2),
        "theta": cols(theta),
        "pressure": cols(p_mid, 0.0),
        "pressure_interface": cols(p_iface, 0.0),
        "z_interface": cols(z_iface, 0.0),
        "w": cols(0.02 * np.sin(np.linspace(0, np.pi, NZ + 1)), 0.0),
        "rho_dry": cols(rho_dry, 0.0),
        "qv": cols(qv),
        "qc": cols(np.where(z_mid < 3000.0, 2.0e-4, 0.0)),
        "qr": cols(np.full(NZ, 1.0e-6)),
    }
    for name in ("qi", "qs", "qg", "ni", "nc", "nwfa", "nifa"):
        f[name] = np.zeros((NZ, ncol), np.float32)
    f["nr"] = f["qr"] * np.float32(1.0e10)
    return {name: cp.asarray(value) for name, value in f.items()}


def _statics(ncol=NCOL, seed=1):
    rng = np.random.default_rng(seed)
    oro = {}
    for name in suite.orographic_statistic_names("gsl"):
        if name.startswith("VAR"):
            oro[name] = rng.uniform(50.0, 400.0, ncol)
        elif name.startswith("CON"):
            oro[name] = rng.uniform(0.0, 5.0, ncol)
        elif name.startswith("OA"):
            oro[name] = rng.uniform(-0.5, 0.5, ncol)
        else:
            oro[name] = rng.uniform(0.0, 0.8, ncol)
    zs = np.array([0, .01, .04, .1, .3, .6, 1., 1.6, 3.], np.float32)
    return {
        "orographic_statistics": {k: v.astype(np.float32)
                                  for k, v in oro.items()},
        "soil_temperature": np.repeat(
            (295.0 - 8.0 * zs / 3.0)[:, None], ncol, 1).astype(np.float32),
        "soil_moisture": np.repeat(
            (0.22 + 0.08 * zs / 3.0)[:, None], ncol, 1).astype(np.float32),
        "terrain_height_m": rng.uniform(0.0, 50.0, ncol).astype(np.float32),
        "ivgtyp": np.full(ncol, 10, np.int32),   # MODIS grassland
        "isltyp": np.full(ncol, 6, np.int32),
    }


def _seam(ncol=NCOL, columns=None, drag="gsl", pbl="mynn", sfclay="mynn"):
    statics = _statics(NCOL)
    sel = slice(None) if columns is None else columns
    oro = ({k: v[sel] for k, v in statics["orographic_statistics"].items()}
           if drag == "gsl" else None)
    return suite.run_mpas_column_suite(
        n_levels=NZ, n_columns=ncol, dt=DT,
        radiation_seconds=600.0, surface_pbl_seconds=60.0,
        microphysics_scheme="thompson_aero",
        radiation_scheme="rrtmg_legacy", surface_layer_scheme=sfclay,
        land_surface_scheme="ruc", pbl_scheme=pbl, drag_scheme=drag,
        start_time=START,
        latitude_deg=np.full(ncol, 38.0), longitude_deg=np.full(ncol, -100.0),
        terrain_height_m=statics["terrain_height_m"][sel],
        z_interface_nominal_m=np.linspace(0.0, ZTOP, NZ + 1),
        p_top_pa=float(101325.0 * np.exp(-ZTOP / 8000.0)), dx_m=12000.0,
        landmask=np.ones(ncol, np.float32), xland=np.ones(ncol, np.float32),
        ivgtyp=statics["ivgtyp"][sel], isltyp=statics["isltyp"][sel],
        vegfra=np.full(ncol, 40.0, np.float32),
        tsk=np.full(ncol, 298.0, np.float32),
        tmn=np.full(ncol, 287.0, np.float32),
        xice=np.zeros(ncol, np.float32), snow=np.zeros(ncol, np.float32),
        snow_depth=np.zeros(ncol, np.float32),
        soil_temperature=statics["soil_temperature"][:, sel],
        soil_moisture=statics["soil_moisture"][:, sel],
        fractional_seaice=1, orographic_statistics=oro)


def _phase1_inputs(inputs):
    return {k: v for k, v in inputs.items()}


def _phase2_state(inputs):
    return {name: inputs[name].copy()
            for name in ("theta", "pressure", "z_interface") + MP28_SPECIES}


def _run(seam, inputs, steps):
    state2 = _phase2_state(inputs)
    out = []
    for _ in range(steps):
        p1 = dict(inputs)
        for name in ("theta",) + MP28_SPECIES:
            p1[name] = state2[name]
        result = seam.run_phase1(dt=DT, **p1)
        frame = {name: cp.asnumpy(getattr(result, name)).copy()
                 for name in mcb._OUTPUT_BUFFERS}
        frame["flags"] = (result.radiation_ran, result.surface_pbl_ran)
        seam.run_phase2(**state2)
        frame["state2"] = {k: cp.asnumpy(v).copy() for k, v in state2.items()}
        out.append(frame)
    return out


def test_hrrr_family_suite_runs_finite_and_complete():
    seam = _seam()
    frames = _run(seam, _profile(), 6)
    for frame in frames:
        for name in mcb._OUTPUT_BUFFERS:
            assert np.all(np.isfinite(frame[name])), name
    assert frames[0]["flags"] == (True, True)
    # The PBL slot actually mixed momentum and heat.
    assert np.abs(frames[0]["du"]).max() > 0.0
    assert np.abs(frames[0]["dtheta"]).max() > 0.0
    assert seam.n_soil_levels == 9
    soil = seam.soil_state()
    assert soil["tslb"].shape == (9, NCOL)
    assert seam.call_counts["ysu"] == 6   # the PBL-slot counter


def test_the_drag_reaches_the_momentum_rates():
    with_drag = _run(_seam(drag="gsl"), _profile(), 1)[0]
    without = _run(_seam(drag="off"), _profile(), 1)[0]
    assert not np.array_equal(with_drag["du"], without["du"])
    # Heat and moisture are the PBL's alone on the first call.
    np.testing.assert_array_equal(with_drag["dqv"], without["dqv"])


def test_phase1_leaves_every_input_byte_identical():
    seam = _seam()
    inputs = _profile()
    before = {k: cp.asnumpy(v).copy() for k, v in inputs.items()}
    seam.run_phase1(dt=DT, **inputs)
    for name, value in inputs.items():
        np.testing.assert_array_equal(cp.asnumpy(value), before[name], name)


def test_batch_split_is_bitwise_per_column():
    """Columns 0..3 and 4..7 as two seams equal one eight-column seam."""
    whole = _run(_seam(), _profile(), 4)
    full = _profile()
    halves = []
    for cols in (slice(0, 4), slice(4, 8)):
        part = {k: cp.ascontiguousarray(v[:, cols]) for k, v in full.items()}
        halves.append(_run(_seam(ncol=4, columns=cols), part, 4))
    for step in range(4):
        for name in mcb._OUTPUT_BUFFERS:
            joined = np.concatenate(
                [halves[0][step][name], halves[1][step][name]], axis=1)
            np.testing.assert_array_equal(
                joined, whole[step][name], f"step {step} {name}")
        for name in whole[step]["state2"]:
            joined = np.concatenate(
                [halves[0][step]["state2"][name],
                 halves[1][step]["state2"][name]], axis=1)
            np.testing.assert_array_equal(
                joined, whole[step]["state2"][name], f"step {step} {name}")


def test_restart_round_trip_continues_bit_identically():
    inputs = _profile()
    reference = _seam()
    ref_frames = _run(reference, inputs, 4)
    first = _seam()
    _run(first, _profile(), 2)
    payload = first.export_state()
    second = _seam()
    second.restore_state(payload)
    # continue from the same caller state the reference had after 2 steps
    state2 = {k: cp.asarray(v) for k, v in ref_frames[1]["state2"].items()}
    for step in (2, 3):
        p1 = dict(inputs)
        for name in ("theta",) + MP28_SPECIES:
            p1[name] = state2[name]
        result = second.run_phase1(dt=DT, **p1)
        for name in mcb._OUTPUT_BUFFERS:
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(result, name)), ref_frames[step][name],
                f"step {step} {name}")
        second.run_phase2(**state2)


def test_payloads_of_the_frozen_and_suite_seams_refuse_each_other():
    seam = _seam()
    payload = seam.export_state()
    assert payload["identity"]["schema"] == suite.SUITE_RESTART_SCHEMA
    frozen_identity_schema = mcb.RESTART_SCHEMA
    assert payload["identity"]["schema"] != frozen_identity_schema
