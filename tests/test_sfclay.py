"""WRF v4.6.1 MM5 surface-layer options 91 (classic) and 1 (revised).

The transcription authorities are ``phys/module_sf_sfclay.F``
(``SFCLAY1D``) and ``phys/physics_mmm/sf_sfclayrev.F90``
(``sf_sfclayrev_run``; ``module_sf_sfclayrev.F`` is its WRF wrapper).
The CUDA implementation is one thread per horizontal column/surface point.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig, load_config


_BASE_CFG = dict(nx=8, ny=4, nz=6, dx=1000.0, dy=1000.0,
                 ztop=6000.0, dt=5.0, run_seconds=0.0)


def _surface_cases(seed=14, ny=6, nx=9):
    """Plausible randomized surface points with every stability branch.

    Input arrays are deliberately float32: the float64 mirror sees exactly
    the values supplied to the FP32 kernel, isolating arithmetic differences.
    Equal positive qv/qsfc makes virtual-temperature stability depend only on
    the imposed ground/air temperature difference; every third point is
    exactly neutral.
    """
    rng = np.random.default_rng(seed)
    shape = (ny, nx)
    t = rng.uniform(285.0, 305.0, shape)
    p = rng.uniform(88000.0, 102000.0, shape)
    qv = rng.uniform(0.002, 0.016, shape)
    delta = np.resize(np.array([-7.0, 0.0, 7.0]), shape)
    # Make the neutral subset exactly neutral in FP32 too.  Unit Exner and a
    # positive moisture value whose virtual correction rounds to one ensure
    # equality-driven WRF regime 3 is exercised without a tolerance rule.
    neutral = delta == 0.0
    p[neutral] = 100000.0
    qv[neutral] = 1.0e-8
    water = (np.arange(ny * nx).reshape(shape) % 5 == 4) & ~neutral
    znt = rng.uniform(0.015, 0.25, shape)
    znt[water] = 1.0e-4
    xland = np.where(water, 2.0, 1.0)
    data = {
        "u": rng.uniform(2.0, 14.0, shape),
        "v": rng.uniform(-5.0, 5.0, shape),
        "t": t,
        "qv": qv,
        "p": p,
        "dz8w": rng.uniform(25.0, 90.0, shape),
        "psfc": p.copy(),
        "tsk": t + delta,
        "znt": znt,
        "pblh": rng.uniform(300.0, 1400.0, shape),
        "mavail": rng.uniform(0.2, 1.0, shape),
        "xland": xland,
        "qsfc": qv.copy(),
        "ust": rng.uniform(0.12, 0.55, shape),
        # Classic SFCLAY diagnoses unstable z/L from the previous T* (MOL)
        # when old u* >= 0.01; a negative previous value is the normal
        # time-stepping state and ensures its unstable psi-table path runs.
        "mol": np.where(delta > 0.0, -0.2,
                        np.where(delta < 0.0, 0.2, 0.0)),
        "hfx": np.zeros(shape),
        "qfx": np.zeros(shape),
        "lakemask": np.zeros(shape),
    }
    return {name: np.ascontiguousarray(value, dtype=np.float32)
            for name, value in data.items()}


def _run_mirror(data, option):
    from gpuwm.verify.npref import np_sfclay

    return np_sfclay(
        data["u"], data["v"], data["t"], data["qv"], data["p"],
        data["dz8w"], data["psfc"], data["tsk"], data["znt"],
        data["pblh"], data["mavail"], data["xland"], option=option,
        qsfc=data["qsfc"], ust=data["ust"], mol=data["mol"],
        hfx=data["hfx"], qfx=data["qfx"], lakemask=data["lakemask"],
        dx=1000.0,
    )


def test_sfclay_config_default_roundtrip_and_validation(tmp_path):
    assert RunConfig(**_BASE_CFG).sf_sfclay_physics == 0

    def write(value):
        path = tmp_path / f"sfclay-{value}.toml"
        path.write_text(textwrap.dedent(f"""
            [grid]
            nx = 8
            ny = 4
            nz = 6
            dx = 1000.0
            dy = 1000.0
            ztop = 6000.0
            [dynamics]
            dt = 5.0
            sf_sfclay_physics = {value}
            [run]
            run_seconds = 0.0
        """))
        return path

    for option in (0, 1, 91):
        assert load_config(write(option)).sf_sfclay_physics == option
    with pytest.raises(ValueError, match="sf_sfclay_physics"):
        load_config(write(2))


@pytest.mark.parametrize("option", [91, 1])
def test_mirror_exercises_stable_neutral_unstable_and_mo_consistency(option):
    from gpuwm.core import constants as c

    data = _surface_cases(ny=1, nx=3)
    old_ust = data["ust"].astype(np.float64)
    out = _run_mirror(data, option)

    # Ground temperatures are air-7 K, air, air+7 K respectively.
    assert out["br"][0, 0] > 0.0
    assert out["br"][0, 1] == 0.0
    assert out["br"][0, 2] < 0.0
    assert int(out["regime"][0, 0]) in ((1, 2) if option == 91 else (1,))
    assert int(out["regime"][0, 1]) == 3
    assert int(out["regime"][0, 2]) == 4

    # WRF's integrated stability corrections: stable <= 0, neutral == 0,
    # unstable >= 0.  Both files constrain z/L to the tabulated/solved range.
    assert out["psim"][0, 0] <= 0.0 and out["psih"][0, 0] <= 0.0
    assert out["psim"][0, 1] == 0.0 and out["psih"][0, 1] == 0.0
    assert out["psim"][0, 2] >= 0.0 and out["psih"][0, 2] >= 0.0
    assert np.max(np.abs(out["zol"])) <= 10.0
    za = 0.5 * data["dz8w"].astype(np.float64)
    consistent = np.ones(out["zol"].shape, dtype=bool)
    if option == 91:
        # WRF's classic regime 1 updates RMOL but leaves inout ZOL untouched.
        consistent = out["regime"] != 1.0
    np.testing.assert_allclose((out["rmol"] * za)[consistent],
                               out["zol"][consistent], rtol=2e-14,
                               atol=2e-14)

    # SFCLAY damps u* by averaging new similarity-theory u* with its old
    # value.  Undo that average and recover k U / (ln(z/z0)-psi_m).
    diagnosed = 2.0 * out["ust"] - old_ust
    np.testing.assert_allclose(diagnosed, 0.4 * out["wspd"] / out["fm"],
                               rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(out["mol"],
                               0.4 * (out["theta_air"]
                                      - out["theta_ground"]) / out["fh"],
                               rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(out["lh"], c.XLV * out["qfx"], rtol=2e-14)
    np.testing.assert_allclose(out["hfx"],
                               out["flhc"] * (out["theta_ground"]
                                              - out["theta_air"]),
                               rtol=2e-14, atol=2e-14)


def test_revised_and_classic_are_distinct_transcriptions():
    data = _surface_cases(ny=2, nx=6)
    classic = _run_mirror(data, 91)
    revised = _run_mirror(data, 1)
    unstable = classic["br"] < 0.0
    stable = classic["br"] > 0.0
    assert np.max(np.abs(classic["psim"][unstable]
                         - revised["psim"][unstable])) > 1.0e-3
    assert np.max(np.abs(classic["psih"][stable]
                         - revised["psih"][stable])) > 1.0e-3


@pytest.mark.parametrize("option", [91, 1])
def test_one_fp32_ulp_temperature_difference_is_not_neutral(option):
    """WRF enters regime 3 only when its directly computed BR is zero."""
    from gpuwm.verify.npref import np_sfclay

    shape = (1, 1)
    full = lambda value: np.full(shape, value, np.float32)
    tsk = np.nextafter(np.float32(300.0), np.float32(np.inf))
    out = np_sfclay(
        full(5.0), full(0.0), full(300.0), full(0.01), full(100000.0),
        full(40.0), full(100000.0), full(tsk), full(0.1), full(800.0),
        full(1.0), full(1.0), option=option, qsfc=full(0.01),
        ust=full(0.2), mol=full(0.0), hfx=full(0.0), qfx=full(0.0),
    )

    assert float(tsk - np.float32(300.0)) == 3.0517578125e-05
    assert out["theta_ground"][0, 0] > out["theta_air"][0, 0]
    assert out["br"][0, 0] < 0.0
    assert int(out["regime"][0, 0]) == 4


def test_classic_strong_stable_preserves_incoming_zol_and_updates_rmol():
    """Option 91 regime 1 updates RMOL but leaves WRF's inout ZOL alone."""
    from gpuwm.core import constants as c
    from gpuwm.verify.npref import np_sfclay

    shape = (1, 1)
    full = lambda value: np.full(shape, value, np.float32)
    incoming_zol = 7.25
    old_ust = 0.2
    old_mol = 0.4
    za = 40.0
    out = np_sfclay(
        full(0.1), full(0.0), full(300.0), full(0.01), full(100000.0),
        full(2.0 * za), full(100000.0), full(290.0), full(0.1),
        full(800.0), full(1.0), full(1.0), option=91, qsfc=full(0.01),
        ust=full(old_ust), mol=full(old_mol), zol=full(incoming_zol),
        hfx=full(0.0), qfx=full(0.0),
    )

    assert out["br"][0, 0] >= 0.2
    assert int(out["regime"][0, 0]) == 1
    assert out["zol"][0, 0] == incoming_zol
    za_over_l = min(0.4 * c.G / 300.0 * za * old_mol / old_ust ** 2,
                    9.999)
    assert out["rmol"][0, 0] == pytest.approx(za_over_l / za)
    assert out["rmol"][0, 0] * za != out["zol"][0, 0]


@pytest.mark.parametrize("option", [91, 1])
def test_coare_style_marine_flux_sanity(option):
    """Canonical 10 m/s, warm/moist sea case stays in published bulk-flux
    scales (COARE-style sanity, not a claim of bitwise COARE equivalence).

    The broad physical gates cover the usual order of magnitude: drag near
    1e-3, u* a few tenths m/s, sensible heat tens W/m2 or less, and latent
    heat tens-to-hundreds W/m2.
    """
    shape = (1, 1)
    full = lambda value: np.full(shape, value, np.float32)
    data = {
        "u": full(10.0), "v": full(0.0), "t": full(300.0),
        "qv": full(0.018), "p": full(101325.0), "dz8w": full(40.0),
        "psfc": full(101325.0), "tsk": full(301.0), "znt": full(1.0e-4),
        "pblh": full(800.0), "mavail": full(1.0), "xland": full(2.0),
        "qsfc": full(0.0), "ust": full(0.35), "mol": full(0.0),
        "hfx": full(0.0), "qfx": full(0.0), "lakemask": full(0.0),
    }
    out = _run_mirror(data, option)
    scalar = lambda name: float(out[name][0, 0])
    assert 0.2 < scalar("ust") < 0.7
    assert 5.0e-4 < scalar("cd") < 3.5e-3
    assert 0.0 < scalar("hfx") < 100.0
    assert 1.0e-5 < scalar("qfx") < 2.0e-4
    assert 25.0 < scalar("lh") < 500.0
    assert 6.0 < scalar("u10") < 12.0


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("option", [91, 1])
def test_sfclay_kernel_matches_float64_mirror_all_regimes(option):
    import cupy as cp

    from gpuwm.core.sfclay import SFCLAY_OUTPUTS, sfclay

    data = _surface_cases(ny=7, nx=11)
    dev = {name: cp.asarray(value) for name, value in data.items()}
    got = sfclay(
        dev["u"], dev["v"], dev["t"], dev["qv"], dev["p"],
        dev["dz8w"], dev["psfc"], dev["tsk"], dev["znt"],
        dev["pblh"], dev["mavail"], dev["xland"], option=option,
        qsfc=dev["qsfc"], ust=dev["ust"], mol=dev["mol"],
        hfx=dev["hfx"], qfx=dev["qfx"], lakemask=dev["lakemask"],
        dx=1000.0,
    )
    ref = _run_mirror(data, option)

    regimes = set(cp.asnumpy(got.regime).astype(int).ravel())
    assert {3, 4}.issubset(regimes)
    assert regimes.intersection({1, 2})
    assert got.ust.dtype == cp.float32
    for name in SFCLAY_OUTPUTS:
        actual = cp.asnumpy(getattr(got, name))
        if name == "regime":
            np.testing.assert_array_equal(actual, ref[name], err_msg=name)
        else:
            scale = max(float(np.max(np.abs(ref[name]))), 1.0e-7)
            np.testing.assert_allclose(actual, ref[name], rtol=5.0e-4,
                                       atol=5.0e-5 * scale, err_msg=name)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("option", [91, 1])
def test_sfclay_kernel_one_fp32_ulp_temperature_difference_is_not_neutral(option):
    import cupy as cp

    from gpuwm.core.sfclay import sfclay

    shape = (1, 1)
    full = lambda value: cp.full(shape, value, cp.float32)
    tsk = np.nextafter(np.float32(300.0), np.float32(np.inf))
    got = sfclay(
        full(5.0), full(0.0), full(300.0), full(0.01), full(100000.0),
        full(40.0), full(100000.0), full(tsk), full(0.1), full(800.0),
        full(1.0), full(1.0), option=option, qsfc=full(0.01),
        ust=full(0.2), mol=full(0.0), hfx=full(0.0), qfx=full(0.0),
    )

    assert float(got.br.get()[0, 0]) < 0.0
    assert int(got.regime.get()[0, 0]) == 4


@pytest.mark.gpu
@requires_gpu
def test_sfclay_kernel_classic_strong_stable_preserves_incoming_zol():
    import cupy as cp

    from gpuwm.core.sfclay import sfclay

    shape = (1, 1)
    full = lambda value: cp.full(shape, value, cp.float32)
    incoming_zol = np.float32(7.25)
    got = sfclay(
        full(0.1), full(0.0), full(300.0), full(0.01), full(100000.0),
        full(80.0), full(100000.0), full(290.0), full(0.1), full(800.0),
        full(1.0), full(1.0), option=91, qsfc=full(0.01),
        ust=full(0.2), mol=full(0.4), zol=full(incoming_zol),
        hfx=full(0.0), qfx=full(0.0),
    )

    assert float(got.br.get()[0, 0]) >= 0.2
    assert int(got.regime.get()[0, 0]) == 1
    assert got.zol.get()[0, 0] == incoming_zol
    assert got.rmol.get()[0, 0] * np.float32(40.0) != incoming_zol


@pytest.mark.gpu
@requires_gpu
def test_sfclay_public_validation():
    import cupy as cp

    from gpuwm.core.sfclay import sfclay

    data = _surface_cases(ny=2, nx=2)
    args = [cp.asarray(data[name]) for name in
            ("u", "v", "t", "qv", "p", "dz8w", "psfc", "tsk", "znt",
             "pblh", "mavail", "xland")]
    with pytest.raises(ValueError, match="option"):
        sfclay(*args, option=2)
    with pytest.raises(ValueError, match="isftcflx"):
        sfclay(*args, option=91, isftcflx=3)
    with pytest.raises(ValueError, match="iz0tlnd"):
        sfclay(*args, option=91, iz0tlnd=3)
