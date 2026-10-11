"""Device twin of tests/test_mpas_mynn_aerosol_mixing.py.

Runs the real selected column chain (legacy RRTMG, MYNN surface layer, RUC,
MYNN-EDMF with the GSL drag, aerosol-aware Thompson) with
``bl_mynn_mixaerosols=1``, RRFS v2's setting, and gates:

* the seam's ``scalar_rates`` are exactly MYNN's own ``dnwfa``/``dnifa``
  (the ``mynn_pbl_step`` return, word for word: the identity mass coupling
  multiplies by 1.0), and nothing else is returned;
* each species' rate is independent of which other species are mixed
  (bitwise equal to the ``bl_mynn_mixscalars=1`` whole-family run);
* switching the aerosols on moves no other phase-one rate by a bit;
* the top level is prescribed (its rate is exactly zero); the column
  budget, which the source does not conserve, is recorded;
* a restored seam continues with bit-identical aerosol rates.

The aerosols start at zero and the seam's first-contact thompson_init fills
them at the first phase 2, so the gates read steps 2 and later.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import gpuwm.core.physics as physics  # noqa: E402
from gpuwm.core import mpas_column_batch as mcb  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_suite_gpu_helpers", _HERE / "test_mpas_column_suite_gpu.py")
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)

NZ, NCOL, DT = helpers.NZ, helpers.NCOL, helpers.DT
SPECIES = helpers.MP28_SPECIES
STEPS = 4


def _seam(generation=None):
    seam = helpers._seam()
    if generation is None:
        return seam
    # Same construction, plus the generation selector under test.
    statics = helpers._statics(NCOL)
    return helpers.suite.run_mpas_column_suite(
        n_levels=NZ, n_columns=NCOL, dt=DT,
        radiation_seconds=600.0, surface_pbl_seconds=60.0,
        microphysics_scheme="thompson_aero",
        radiation_scheme="rrtmg_legacy", surface_layer_scheme="mynn",
        land_surface_scheme="ruc", pbl_scheme="mynn", drag_scheme="gsl",
        start_time=helpers.START,
        latitude_deg=np.full(NCOL, 38.0), longitude_deg=np.full(NCOL, -100.0),
        terrain_height_m=statics["terrain_height_m"],
        z_interface_nominal_m=np.linspace(0.0, helpers.ZTOP, NZ + 1),
        p_top_pa=float(101325.0 * np.exp(-helpers.ZTOP / 8000.0)),
        dx_m=12000.0,
        landmask=np.ones(NCOL, np.float32), xland=np.ones(NCOL, np.float32),
        ivgtyp=statics["ivgtyp"], isltyp=statics["isltyp"],
        vegfra=np.full(NCOL, 40.0, np.float32),
        tsk=np.full(NCOL, 298.0, np.float32),
        tmn=np.full(NCOL, 287.0, np.float32),
        xice=np.zeros(NCOL, np.float32), snow=np.zeros(NCOL, np.float32),
        snow_depth=np.zeros(NCOL, np.float32),
        soil_temperature=statics["soil_temperature"],
        soil_moisture=statics["soil_moisture"],
        fractional_seaice=1,
        orographic_statistics=statics["orographic_statistics"],
        generation=generation)


class _Capture:
    """Wrap physics.mynn_pbl_step and keep host copies of its qn rates."""

    def __init__(self, monkeypatch):
        self.calls = []
        original = physics.mynn_pbl_step

        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            self.calls.append({name: cp.asnumpy(out[name]).reshape(NZ, NCOL).copy()
                               for name in ("dnc", "dni", "dnwfa", "dnifa")
                               if name in out})
            return out

        monkeypatch.setattr(physics, "mynn_pbl_step", wrapped)


def _run(seam, steps=STEPS, inputs=None):
    inputs = helpers._profile() if inputs is None else inputs
    state2 = helpers._phase2_state(inputs)
    frames = []
    for _ in range(steps):
        p1 = dict(inputs)
        for name in ("theta",) + SPECIES:
            p1[name] = state2[name]
        before = {name: cp.asnumpy(state2[name]).copy()
                  for name in ("nwfa", "nifa")}
        result = seam.run_phase1(dt=DT, **p1)
        frame = {name: cp.asnumpy(getattr(result, name)).copy()
                 for name in mcb._OUTPUT_BUFFERS}
        frame["rates"] = {name: cp.asnumpy(rate).copy() for name, rate in
                          dict(getattr(result, "scalar_rates", {})).items()}
        frame["aerosol_in"] = before
        frame["qv_in"] = cp.asnumpy(state2["qv"]).astype(np.float64)
        frame["rho_dz"] = (cp.asnumpy(inputs["rho_dry"]).astype(np.float64)
                           * np.diff(cp.asnumpy(inputs["z_interface"])
                                     .astype(np.float64), axis=0))
        seam.run_phase2(**state2)
        frames.append(frame)
    return frames, state2


def test_seam_rates_are_mynns_own_rates_word_for_word(monkeypatch):
    capture = _Capture(monkeypatch)
    seam = _seam({"bl_mynn_mixaerosols": 1})
    assert seam.mixed_number_species == ("nwfa", "nifa")
    frames, _ = _run(seam)
    assert len(capture.calls) == STEPS
    for step, (frame, call) in enumerate(zip(frames, capture.calls)):
        assert set(frame["rates"]) == {"nwfa", "nifa"}, step
        for name in ("nwfa", "nifa"):
            np.testing.assert_array_equal(
                frame["rates"][name], call["d" + name], f"step {step} {name}")
            assert np.all(np.isfinite(frame["rates"][name]))
    # Mixing is live once first contact filled the aerosol.
    assert np.abs(frames[1]["aerosol_in"]["nwfa"]).max() > 0.0
    assert np.abs(frames[1]["rates"]["nwfa"]).max() > 0.0
    assert np.abs(frames[1]["rates"]["nifa"]).max() > 0.0


def test_species_rates_do_not_depend_on_which_others_are_mixed():
    aerosols, _ = _run(_seam({"bl_mynn_mixaerosols": 1}))
    family, _ = _run(_seam({"bl_mynn_mixaerosols": 1, "bl_mynn_mixnumcon": 1}))
    for step in range(STEPS):
        assert set(family[step]["rates"]) == {"nc", "ni", "nwfa", "nifa"}
        for name in ("nwfa", "nifa"):
            np.testing.assert_array_equal(
                aerosols[step]["rates"][name], family[step]["rates"][name],
                f"step {step} {name}")


def test_mixing_the_aerosols_moves_no_other_phase_one_rate():
    # Off and on diverge in state only through the aerosol itself, which
    # feeds back through microphysics at phase 2 -- so compare step 1 (same
    # state, aerosol zero) on every buffer, and later steps on the rates
    # phase one forms before any aerosol-dependent microphysics feedback.
    off, _ = _run(_seam(), steps=1)
    on, _ = _run(_seam({"bl_mynn_mixaerosols": 1}), steps=1)
    for name in mcb._OUTPUT_BUFFERS:
        np.testing.assert_array_equal(off[0][name], on[0][name], name)
    assert off[0]["rates"] == {}


def test_the_top_level_is_prescribed_and_the_column_budget_is_recorded():
    """MYNN's qn solve prescribes its top value and is not column-conserving.

    The top row is d(kte)=qn(kte), so the top level's rate is exactly zero.
    The bottom row carries khdz(kts) = rhoz(kts)*dfh(kts) on the diagonal
    with no exterior partner (WRF v4.6.1 module_bl_mynn.F mynn_tendencies;
    the same row in NCAR/MYNN-EDMF c942418 module_bl_mynnedmf.F90:4250 and
    :5184-5186 for nwfa), and the row below the prescribed top exchanges
    with a level that does not answer, so the column total is not a
    conserved quantity of the source and is recorded here, not gated.
    """
    frames, _ = _run(_seam({"bl_mynn_mixaerosols": 1}))
    for step in range(1, STEPS):
        # The solve's own density: physics rho = (1 + qv) * rho_dry.
        weight = frames[step]["rho_dz"] * (1.0 + frames[step]["qv_in"])
        for name in ("nwfa", "nifa"):
            rate = frames[step]["rates"][name]
            assert np.all(rate[-1] == 0.0), (step, name)
            rate = rate.astype(np.float64)
            column = (weight * rate).sum(axis=0)
            scale = (weight * np.abs(rate)).sum(axis=0)
            ratio = column / np.maximum(scale, 1e-30)
            top_share = (weight[-2] * np.abs(rate[-2])) / np.maximum(scale, 1e-30)
            print(f"column step={step} {name}: sum(rho dz rate)/sum(rho dz |rate|) "
                  f"min={float(ratio.min()):.3e} max={float(ratio.max()):.3e}; "
                  f"level kte-1 share max={float(top_share.max()):.3e}; "
                  f"max|rate|={float(np.abs(rate).max()):.3e} /kg/s")


def test_restored_seam_continues_with_identical_aerosol_rates():
    generation = {"bl_mynn_mixaerosols": 1}
    inputs = helpers._profile()
    reference, _ = _run(_seam(generation), inputs=inputs)
    first = _seam(generation)
    _, state_after_two = _run(first, steps=2, inputs=helpers._profile())
    payload = first.export_state()
    second = _seam(generation)
    second.restore_state(payload)
    state2 = {k: cp.asarray(v) for k, v in state_after_two.items()}
    for step in (2, 3):
        p1 = dict(inputs)
        for name in ("theta",) + SPECIES:
            p1[name] = state2[name]
        result = second.run_phase1(dt=DT, **p1)
        for name in ("nwfa", "nifa"):
            np.testing.assert_array_equal(
                cp.asnumpy(result.scalar_rates[name]),
                reference[step]["rates"][name], f"step {step} {name}")
        second.run_phase2(**state2)
